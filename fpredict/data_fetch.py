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
from datetime import date
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
    return f"{config.BASE_URL}/{season_code(start_year)}/{league_code}.csv"


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
    """football-data tarihini ISO 'YYYY-MM-DD' string'ine çevirir."""
    if pd.isna(raw):
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(str(raw).strip(), format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # Son çare: pandas'ın esnek ayrıştırıcısı (gün-önce varsayımıyla)
    try:
        return pd.to_datetime(str(raw).strip(), dayfirst=True).strftime("%Y-%m-%d")
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


# --------------------------------------------------------------------------- #
# İndirme
# --------------------------------------------------------------------------- #
def download_csv(url: str, timeout: int = config.HTTP_TIMEOUT) -> bytes:
    """Tek bir CSV'yi indirir. Ağ hatalarını DataFetchError'a çevirir."""
    import requests  # yerel import: test ederken ağ bağımlılığını izole eder

    try:
        resp = requests.get(
            url, timeout=timeout, headers={"User-Agent": config.USER_AGENT}
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise DataFetchError(
            f"İndirme başarısız ({url}): {exc}. "
            "İnternet bağlantınızı kontrol edin veya CSV'yi elle yükleyin."
        ) from exc
    if not resp.content or len(resp.content) < 50:
        raise DataFetchError(f"İndirilen dosya boş görünüyor: {url}")
    return resp.content


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
    _, code = config.LEAGUES[league_key]
    years = recent_seasons(seasons_back)

    total_inserted = 0
    seasons_ok, seasons_failed, errors = [], [], []
    for i, year in enumerate(years):
        sc = season_code(year)
        if progress:
            progress(f"{code} {sc} indiriliyor…", i / len(years))
        try:
            content = download_csv(csv_url(code, year))
            rows = parse_csv_bytes(content, code, season=sc)
            total_inserted += db.upsert_matches(rows, db_path=db_path)
            seasons_ok.append(sc)
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
    }
