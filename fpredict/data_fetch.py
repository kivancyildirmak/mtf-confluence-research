"""football-data.co.uk üzerinden geçmiş sonuç + oran indirme ve ayrıştırma.

URL kalıbı:
    https://www.football-data.co.uk/mmz4281/{SSSS}/{KOD}.csv
burada {SSSS} sezon kodu (ör. 2023-24 -> "2324"), {KOD} lig kodu (E0, T1, ...).

Tasarım ilkeleri
----------------
* Ağ çağrıları try/except ile sarılır; hata mesajları kullanıcıya anlaşılır
  biçimde döner (DataFetchError).
* İndirme başarısızsa çağıran taraf elle CSV yükleme seçeneğine düşebilsin diye
  `parse_csv_bytes` fonksiyonu bayt/dosya girişinden bağımsız çalışır.
* Ayrıştırma football-data.co.uk'un tarih (dd/mm/yy veya dd/mm/yyyy) ve
  eksik/oynak sütun biçimlerine dayanıklıdır.
"""
from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config, db
from .name_matching import normalize


class DataFetchError(RuntimeError):
    """İndirme veya ayrıştırma sırasında oluşan, kullanıcıya gösterilebilir hata."""


# --------------------------------------------------------------------------- #
# Sezon kodu yardımcıları
# --------------------------------------------------------------------------- #
def season_code(start_year: int) -> str:
    """2023 -> '2324' (2023-24 sezonu)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def recent_seasons(n: int, today: date | None = None) -> list[int]:
    """En yeni n sezonun başlangıç yıllarını döndürür.

    Futbol sezonu genellikle Temmuz/Ağustos başlar; Temmuz'dan önce (Oca-Haz)
    içinde bulunulan sezon hâlâ bir önceki yılda başlamıştır.
    """
    today = today or date.today()
    current_start = today.year if today.month >= 7 else today.year - 1
    return [current_start - i for i in range(n)]


def csv_url(league_code: str, start_year: int) -> str:
    """'main' biçim: sezon başına bir dosya."""
    return f"{config.BASE_URL}/{season_code(start_year)}/{league_code}.csv"


def extra_csv_url(league_code: str) -> str:
    """'extra' biçim: tüm sezonlar tek dosyada."""
    return f"{config.EXTRA_BASE_URL}/{league_code}.csv"


# --------------------------------------------------------------------------- #
# CSV ayrıştırma
# --------------------------------------------------------------------------- #
def _first_present(row: pd.Series, columns: Iterable[str]):
    """Satırda mevcut olan ilk (boş olmayan) sütun değerini döndürür."""
    for col in columns:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def _parse_date(raw) -> str | None:
    """football-data tarihini ISO 'YYYY-MM-DD' string'ine çevirir.

    Biçimler açık açık denenir, çünkü esnek ayrıştırıcı `dayfirst=True` ile
    ISO tarihleri YYYY-DD-MM gibi okuyup gün/ayı SESSİZCE takas eder
    (ör. '2023-09-01' -> 1 Eylül yerine 9 Ocak). Bu, zaman ağırlığını ve
    backtest sıralamasını bozar; bu yüzden ISO biçimi listede önce gelir.
    """
    if pd.isna(raw):
        return None
    text = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(text, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # Son çare: yalnızca gün/ay takası riski olmayan durumlar için esnek ayrıştırma.
    # (ISO biçim yukarıda yakalandığı için burada gün-önce varsayımı güvenlidir.)
    try:
        return pd.to_datetime(text, dayfirst=True).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def parse_csv_bytes(raw: bytes | str, league_code: str, season: str | None = None) -> list[dict]:
    """Bir football-data CSV içeriğini normalize maç kayıtlarına çevirir.

    Bilinmeyen/eksik oran sütunları None olarak bırakılır. Skoru olmayan
    (henüz oynanmamış) satırlar atlanır.
    """
    if isinstance(raw, bytes):
        buf = io.BytesIO(raw)
    else:
        buf = io.StringIO(raw)
    try:
        df = pd.read_csv(buf, encoding="latin-1", on_bad_lines="skip")
    except Exception as exc:  # pragma: no cover - bozuk dosya
        raise DataFetchError(f"CSV ayrıştırılamadı: {exc}") from exc

    if "HomeTeam" not in df.columns or "FTHG" not in df.columns:
        raise DataFetchError(
            "Beklenen sütunlar bulunamadı (HomeTeam/FTHG). "
            "Bu dosya football-data.co.uk maç CSV'si olmayabilir."
        )

    rows: list[dict] = []
    for _, r in df.iterrows():
        iso = _parse_date(_first_present(r, ["Date"]))
        home = r.get("HomeTeam")
        away = r.get("AwayTeam")
        if not iso or pd.isna(home) or pd.isna(away):
            continue
        hg = r.get("FTHG")
        ag = r.get("FTAG")
        if pd.isna(hg) or pd.isna(ag):
            continue  # oynanmamış maç
        try:
            hg, ag = int(hg), int(ag)
        except (ValueError, TypeError):
            continue

        home, away = str(home).strip(), str(away).strip()
        rows.append({
            "match_uid": db.make_uid(league_code, iso, home, away),
            "league": league_code,
            "season": season,
            "date": iso,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": r.get("FTR") if pd.notna(r.get("FTR")) else ("H" if hg > ag else "A" if ag > hg else "D"),
            # Oranlar: önce Bet365, yoksa piyasa ortalaması/diğer sağlayıcılar
            "odds_h": _num(_first_present(r, ["B365H", "AvgH", "BbAvH", "PSH", "PSCH"])),
            "odds_d": _num(_first_present(r, ["B365D", "AvgD", "BbAvD", "PSD", "PSCD"])),
            "odds_a": _num(_first_present(r, ["B365A", "AvgA", "BbAvA", "PSA", "PSCA"])),
            "odds_over25": _num(_first_present(r, ["B365>2.5", "Avg>2.5", "BbAv>2.5", "P>2.5"])),
            "odds_under25": _num(_first_present(r, ["B365<2.5", "Avg<2.5", "BbAv<2.5", "P<2.5"])),
        })
    return rows


def _num(val):
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_extra_csv_bytes(raw: bytes | str, league_code: str) -> list[dict]:
    """football-data.co.uk '/new/{KOD}.csv' (extra) biçimini ayrıştırır.

    Bu biçim İskandinav ligleri, Polonya, Avusturya, İsviçre vb. için kullanılır
    ve 'main' biçimden farklıdır:

        main :  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H ...
        extra:  Date, Home,     Away,     HG,   AG,   Res, PH/AvgH ...

    Ayrıca tek dosyada tüm sezonlar bulunur ve bir 'Season' sütunu vardır
    (İskandinav ligleri takvim yılı olduğu için "2025" gibi tek yıl olabilir).
    """
    if isinstance(raw, bytes):
        buf = io.BytesIO(raw)
    else:
        buf = io.StringIO(raw)
    try:
        df = pd.read_csv(buf, encoding="latin-1", on_bad_lines="skip")
    except Exception as exc:  # pragma: no cover - bozuk dosya
        raise DataFetchError(f"CSV ayrıştırılamadı: {exc}") from exc

    if "Home" not in df.columns or "HG" not in df.columns:
        raise DataFetchError(
            "Beklenen sütunlar bulunamadı (Home/HG). "
            "Bu dosya football-data.co.uk 'extra' lig CSV'si olmayabilir."
        )

    rows: list[dict] = []
    for _, r in df.iterrows():
        iso = _parse_date(_first_present(r, ["Date"]))
        home = r.get("Home")
        away = r.get("Away")
        if not iso or pd.isna(home) or pd.isna(away):
            continue
        hg, ag = r.get("HG"), r.get("AG")
        if pd.isna(hg) or pd.isna(ag):
            continue  # oynanmamış maç
        try:
            hg, ag = int(hg), int(ag)
        except (ValueError, TypeError):
            continue

        home, away = str(home).strip(), str(away).strip()
        season = r.get("Season")
        season = str(season).strip() if pd.notna(season) else None

        rows.append({
            "match_uid": db.make_uid(league_code, iso, home, away),
            "league": league_code,
            "season": season,
            "date": iso,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": r.get("Res") if pd.notna(r.get("Res")) else (
                "H" if hg > ag else "A" if ag > hg else "D"
            ),
            # extra biçimde Bet365 sütunu yok: Pinnacle (P*) / ortalama (Avg*) kullan
            "odds_h": _num(_first_present(r, ["AvgH", "PH", "MaxH", "B365H"])),
            "odds_d": _num(_first_present(r, ["AvgD", "PD", "MaxD", "B365D"])),
            "odds_a": _num(_first_present(r, ["AvgA", "PA", "MaxA", "B365A"])),
            "odds_over25": _num(_first_present(r, ["Avg>2.5", "P>2.5", "Max>2.5"])),
            "odds_under25": _num(_first_present(r, ["Avg<2.5", "P<2.5", "Max<2.5"])),
        })
    return rows


def parse_any_csv_bytes(raw: bytes | str, league_code: str, season: str | None = None) -> list[dict]:
    """Biçimi otomatik algılayarak ayrıştırır (elle CSV yüklemede kullanışlı)."""
    head = raw[:2000].decode("latin-1", "ignore") if isinstance(raw, bytes) else raw[:2000]
    first_line = head.splitlines()[0] if head.splitlines() else ""
    if "HomeTeam" in first_line:
        return parse_csv_bytes(raw, league_code, season=season)
    if "Home" in first_line:
        return parse_extra_csv_bytes(raw, league_code)
    raise DataFetchError(
        "CSV biçimi tanınamadı. football-data.co.uk maç CSV'si bekleniyor "
        "(HomeTeam/FTHG veya Home/HG sütunları)."
    )


# --------------------------------------------------------------------------- #
# İndirme
# --------------------------------------------------------------------------- #
def classify_network_error(exc) -> str:
    """Ağ hatasını kullanıcıya anlamlı gelecek kısa bir teşhise çevirir.

    Amaç: "Max retries exceeded" gibi ham yığın izleri yerine, kullanıcının
    ne yapması gerektiğini anlatan bir sebep göstermek.
    """
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "certificate" in low or "sslerror" in low or "ssl:" in low:
        return (
            "TLS sertifika doğrulaması başarısız (self-signed certificate). "
            "Bağlantı araya girilerek yönlendiriliyor olabilir — genellikle "
            "İSS/DNS engeli veya antivirüsün HTTPS taraması buna yol açar."
        )
    if "timed out" in low or "timeout" in low:
        return (
            "Bağlantı zaman aşımına uğradı — sunucuya paket gidiyor ama yanıt "
            "dönmüyor. Genellikle ağ seviyesinde engelleme belirtisidir."
        )
    if "reset" in low or "10054" in low or "aborted" in low:
        return (
            "Bağlantı karşı taraftan zorla kapatıldı (reset). Genellikle ağ "
            "seviyesinde engelleme belirtisidir."
        )
    if "name or service not known" in low or "getaddrinfo" in low or "nodename" in low:
        return "Alan adı çözümlenemedi (DNS hatası)."
    return text


def download_csv(
    url: str,
    timeout: int = config.HTTP_TIMEOUT,
    retries: int = config.HTTP_RETRIES,
) -> bytes:
    """Tek bir CSV'yi indirir; geçici hatalarda yeniden dener.

    Ağ hatalarını, sebebini açıklayan bir DataFetchError'a çevirir.
    """
    import time

    import requests  # yerel import: test ederken ağ bağımlılığını izole eder

    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.get(
                url, timeout=timeout, headers={"User-Agent": config.USER_AGENT}
            )
            resp.raise_for_status()
            if not resp.content or len(resp.content) < 50:
                raise DataFetchError(f"İndirilen dosya boş görünüyor: {url}")
            return resp.content
        except requests.exceptions.HTTPError as exc:
            # 404 vb. yeniden denemeye değmez (ör. sezon henüz yayınlanmamış)
            status = exc.response.status_code if exc.response is not None else "?"
            raise DataFetchError(f"Sunucu {status} döndü: {url}") from exc
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s …

    raise DataFetchError(
        f"İndirme başarısız ({url}) — {retries} deneme sonunda. "
        f"Sebep: {classify_network_error(last_exc)}"
    )


def download_with_fallback(urls: list[tuple[str, str]]) -> tuple[bytes, str]:
    """Sırayla birden çok kaynağı dener; ilk başarılı olanı döndürür.

    Args:
        urls: [(etiket, url), ...] — örn. [("football-data.co.uk", ...), ("ayna", ...)]

    Returns:
        (içerik, kullanılan_etiket)

    Raises:
        DataFetchError: hepsi başarısızsa, her kaynağın sebebiyle birlikte.
    """
    problems = []
    for label, url in urls:
        try:
            return download_csv(url), label
        except DataFetchError as exc:
            problems.append(f"{label}: {exc}")
    raise DataFetchError(" | ".join(problems))


def update_league(
    league_key: str,
    seasons_back: int = config.DEFAULT_SEASONS_BACK,
    db_path=None,
    progress=None,
) -> dict:
    """Bir ligin son `seasons_back` sezonunu indirip cache'e yazar.

    Args:
        league_key: config.LEAGUES anahtarı (ör. "T1").
        progress: opsiyonel callback(msg: str, frac: float) — arayüz için.

    Returns:
        {"league", "inserted", "seasons_ok", "seasons_failed", "errors"}
    """
    if league_key not in config.LEAGUES:
        raise DataFetchError(f"Bilinmeyen lig: {league_key}")
    info = config.LEAGUES[league_key]
    code = info.code

    # 'extra' ligler tek dosyada gelir; sezon döngüsü yok
    if info.source == "extra":
        return _update_extra_league(league_key, code, db_path=db_path, progress=progress)

    years = recent_seasons(seasons_back)

    total_inserted = 0
    seasons_ok, seasons_failed, errors = [], [], []
    sources_used = set()
    for i, year in enumerate(years):
        sc = season_code(year)
        if progress:
            progress(f"{code} {sc} indiriliyor…", i / len(years))

        # Önce asıl kaynak, olmazsa (varsa) ayna
        candidates = [("football-data.co.uk", csv_url(code, year))]
        mirror = config.mirror_url(code, sc)
        if mirror:
            candidates.append(("GitHub aynası", mirror))

        try:
            content, used = download_with_fallback(candidates)
            rows = parse_csv_bytes(content, code, season=sc)
            total_inserted += db.upsert_matches(rows, db_path=db_path)
            seasons_ok.append(sc)
            sources_used.add(used)
        except DataFetchError as exc:
            seasons_failed.append(sc)
            errors.append(str(exc))

    if seasons_ok:
        db.touch_updated(code, db_path=db_path)
    if progress:
        progress("Tamamlandı.", 1.0)

    return {
        "league": league_key,
        "inserted": total_inserted,
        "seasons_ok": seasons_ok,
        "seasons_failed": seasons_failed,
        "errors": errors,
        "sources_used": sorted(sources_used),
    }


def ensure_archive(progress=None, force: bool = False) -> Path:
    """Birleşik arşiv CSV'sini indirir ve yerelde saklar (~43 MB, tek sefer).

    Yerel kopya `ARCHIVE_MAX_AGE_DAYS` günden yeniyse yeniden indirilmez.

    Returns:
        Yerel arşiv dosyasının yolu.

    Raises:
        DataFetchError: indirilemezse ve yerel kopya da yoksa.
    """
    config.ensure_app_dir()
    path = config.ARCHIVE_CACHE

    if path.exists() and not force:
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < config.ARCHIVE_MAX_AGE_DAYS:
            return path

    if progress:
        progress("Arşiv indiriliyor (~43 MB, yalnızca ilk seferde)…", 0.1)
    try:
        content = download_csv(config.ARCHIVE_URL, timeout=180)
    except DataFetchError:
        if path.exists():
            return path  # eski kopya yenisinden iyidir
        raise
    path.write_bytes(content)
    return path


def parse_archive_for_league(archive_path: Path, league_key: str) -> list[dict]:
    """Arşivden tek bir ligin maçlarını normalize kayıtlara çevirir.

    Arşiv sütunları football-data.co.uk'tan farklı adlandırılmıştır:
        MatchDate, HomeTeam, AwayTeam, FTHome, FTAway, FTResult,
        OddHome/OddDraw/OddAway, Over25/Under25
    """
    division = config.archive_division(league_key)
    if not division:
        raise DataFetchError(f"{league_key} arşivde bulunmuyor.")

    cols = [
        "Division", "MatchDate", "HomeTeam", "AwayTeam",
        "FTHome", "FTAway", "FTResult",
        "OddHome", "OddDraw", "OddAway", "Over25", "Under25",
    ]
    try:
        df = pd.read_csv(archive_path, usecols=cols, low_memory=False)
    except Exception as exc:
        raise DataFetchError(f"Arşiv okunamadı: {exc}") from exc

    df = df[df["Division"] == division]
    if df.empty:
        raise DataFetchError(f"Arşivde '{division}' için maç bulunamadı.")

    code = config.LEAGUES[league_key].code
    rows: list[dict] = []
    for _, r in df.iterrows():
        iso = _parse_date(r.get("MatchDate"))
        home, away = r.get("HomeTeam"), r.get("AwayTeam")
        hg, ag = r.get("FTHome"), r.get("FTAway")
        if not iso or pd.isna(home) or pd.isna(away) or pd.isna(hg) or pd.isna(ag):
            continue
        try:
            hg, ag = int(hg), int(ag)
        except (ValueError, TypeError):
            continue
        home, away = str(home).strip(), str(away).strip()
        rows.append({
            "match_uid": db.make_uid(code, iso, home, away),
            "league": code,
            "season": iso[:4],
            "date": iso,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": r.get("FTResult") if pd.notna(r.get("FTResult")) else (
                "H" if hg > ag else "A" if ag > hg else "D"
            ),
            "odds_h": _num(r.get("OddHome")),
            "odds_d": _num(r.get("OddDraw")),
            "odds_a": _num(r.get("OddAway")),
            "odds_over25": _num(r.get("Over25")),
            "odds_under25": _num(r.get("Under25")),
        })
    return rows


def update_from_archive(league_key: str, db_path=None, progress=None) -> dict:
    """Bir ligi arşiv kaynağından günceller (football-data.co.uk erişilemezken).

    Returns:
        update_league ile aynı biçimde sonuç sözlüğü; ayrıca 'last_match_date'
        ve 'stale_days' alanlarıyla verinin ne kadar eski olduğunu bildirir.
    """
    code = config.LEAGUES[league_key].code
    path = ensure_archive(progress=progress)
    if progress:
        progress("Arşiv ayrıştırılıyor…", 0.6)
    rows = parse_archive_for_league(path, league_key)
    inserted = db.upsert_matches(rows, db_path=db_path)
    db.touch_updated(code, db_path=db_path)
    if progress:
        progress("Tamamlandı.", 1.0)

    last_date = max((r["date"] for r in rows), default=None)
    stale_days = None
    if last_date:
        stale_days = (date.today() - date.fromisoformat(last_date)).days

    return {
        "league": league_key,
        "inserted": inserted,
        "seasons_ok": sorted({r["season"] for r in rows})[-5:],
        "seasons_failed": [],
        "errors": [],
        "sources_used": ["Arşiv (GitHub)"],
        "last_match_date": last_date,
        "stale_days": stale_days,
    }


def _update_extra_league(league_key: str, code: str, db_path=None, progress=None) -> dict:
    """'extra' biçimli ligi (tek dosya, tüm sezonlar) indirip cache'e yazar."""
    url = extra_csv_url(code)
    if progress:
        progress(f"{code} indiriliyor (tüm sezonlar tek dosyada)…", 0.2)
    try:
        content = download_csv(url)
        rows = parse_extra_csv_bytes(content, code)
        inserted = db.upsert_matches(rows, db_path=db_path)
    except DataFetchError as exc:
        if progress:
            progress("Başarısız.", 1.0)
        return {
            "league": league_key,
            "inserted": 0,
            "seasons_ok": [],
            "seasons_failed": ["tümü"],
            "errors": [str(exc)],
            "sources_used": [],
        }

    db.touch_updated(code, db_path=db_path)
    if progress:
        progress("Tamamlandı.", 1.0)

    seasons = sorted({r["season"] for r in rows if r.get("season")})
    return {
        "league": league_key,
        "inserted": inserted,
        "seasons_ok": seasons or ["tümü"],
        "seasons_failed": [],
        "errors": [],
        "sources_used": ["football-data.co.uk"],
    }
