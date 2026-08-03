#!/usr/bin/env python3
"""Binance'ten geçmiş 1 dakikalık bar indirici (PROTOTİP veri kaynağı).

**NEDEN?** Paribu'da tarihsel mum ucu yoktur; poll-forward toplayıcı
(:mod:`research.tools.collect_paribu`) geçmişi ancak bugünden itibaren
biriktirir ve hattın varsayılan ölçeğine ulaşması ~45 gün sürer. Bu araç, o
süreyi beklemeden hattı **gerçek** kripto verisiyle sınamak için prototip veri
sağlar.

**DÜRÜSTLÜK UYARISI — bu veri Paribu DEĞİLDİR.**
Binance BTCTRY/BTCUSDT ile Paribu BTC_TL farklı borsalardır: farklı likidite,
farklı spread, farklı TL primi, farklı mikroyapı ve farklı icra koşulları.
Burada elde edilen hiçbir sonuç Paribu için geçerli sayılamaz. Bu veri
yalnızca şunun içindir:

* hattın gerçek (sentetik olmayan) veriyle uçtan uca çalıştığını görmek,
* özellik/etiket parametrelerini kabaca kalibre etmek.

Canlıya geçmeden önce her şey Paribu'nun kendi verisiyle **yeniden
doğrulanmalıdır** (README bölüm 10.4).

Anahtar/HMAC YOKTUR — Binance public market data anahtarsızdır.

Kaynaklar (tercih sırasıyla):

1. **Toplu döküm** ``data.binance.vision`` — aylık/günlük 1m klines CSV zip.
   Uzun geçmiş için çok daha verimli (tek istekte bir ay).
2. **REST** ``api.binance.com/api/v3/klines`` — 1000'lik sayfalarla ileri
   sayfalama. Toplu döküme erişilemezse devreye girer.

Kullanım::

    python -m research.tools.fetch_binance                 # ~60 gun, otomatik sembol
    python -m research.tools.fetch_binance --days 90
    python -m research.tools.fetch_binance --symbol BTCUSDT --source rest
    python -m research.tools.fetch_binance --probe-only    # sadece olc, indirme
"""

from __future__ import annotations

import argparse
import io
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from ..config import CONFIG
from ..data import OHLCV_COLUMNS, drop_unclosed_bar, validate_ohlcv

#: Binance kline satırlarının sabit sütun düzeni (REST ve CSV aynıdır).
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)

#: Denenecek semboller: önce TL paritesi, yetersizse USDT paritesi.
DEFAULT_SYMBOLS: tuple[str, ...] = ("BTCTRY", "BTCUSDT")

BULK_BASE = "https://data.binance.vision/data/spot"
REST_URL = "https://api.binance.com/api/v3/klines"
USER_AGENT = "paribu-research-fetch/1.0"
TIMEOUT = 30.0

#: Bir sembolün "kullanılabilir" sayılması için gereken asgari doluluk oranı
#: (gerçek bar sayısı / beklenen bar sayısı). Altındaysa ince kabul edilir.
MIN_COVERAGE = 0.60


def _log(msg: str) -> None:
    """Zaman damgalı ilerleme çıktısı (UTC)."""
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# HTTP (ince, enjekte edilebilir katman)
# --------------------------------------------------------------------------- #


def http_get(url: str, timeout: float = TIMEOUT) -> bytes:
    """Tek bir GET isteği atar ve ham baytları döndürür.

    Args:
        url: İstenecek adres.
        timeout: Zaman aşımı (saniye).

    Returns:
        Yanıt gövdesi.

    Raises:
        urllib.error.HTTPError: 4xx/5xx yanıtlarda (404 = kaynak yok demektir).
        urllib.error.URLError: Ağ hatasında.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
        return r.read()


def http_get_retry(
    url: str,
    retries: int = 4,
    timeout: float = TIMEOUT,
    getter: Callable[[str, float], bytes] = http_get,
) -> bytes:
    """Geri çekilmeli GET. 404'ü YENİDEN DENEMEZ (kaynak yoktur, hata değil).

    Args:
        url: İstenecek adres.
        retries: Azami deneme sayısı.
        timeout: Zaman aşımı.
        getter: Test için enjekte edilebilen alt seviye getirici.

    Returns:
        Yanıt gövdesi.

    Raises:
        urllib.error.HTTPError: 404'te hemen, diğerlerinde denemeler bitince.
    """
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return getter(url, timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise  # kaynak yok; üst katman zaten yedeğe düşecek
            # 429/418 = rate limit: sunucunun istediği kadar bekle.
            wait = float(e.headers.get("Retry-After", 0) or 0) if e.headers else 0.0
            last = e
            time.sleep(max(wait, delay))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay)
        delay = min(delay * 2, 30.0)
        if attempt < retries:
            _log(f"  yeniden deneniyor ({attempt}/{retries}): {url.rsplit('/', 1)[-1]}")
    raise last if last else RuntimeError(f"İstek başarısız: {url}")


# --------------------------------------------------------------------------- #
# Ayrıştırma
# --------------------------------------------------------------------------- #


def infer_epoch_unit(value: float | int) -> str:
    """Epoch zaman damgasının birimini BÜYÜKLÜĞÜNDEN çıkarır.

    Binance bazı dökümlerde milisaniye, bazılarında mikrosaniye kullanır. Birimi
    varsaymak, tarihleri 1000 kat kaydırıp veriyi sessizce çöpe çevirir; bu
    yüzden ölçüyoruz.

    Args:
        value: Örnek zaman damgası.

    Returns:
        ``"s"``, ``"ms"``, ``"us"`` veya ``"ns"``.
    """
    a = abs(float(value))
    if a >= 1e17:
        return "ns"
    if a >= 1e14:
        return "us"
    if a >= 1e11:
        return "ms"
    return "s"


def klines_to_frame(rows: Iterable[Iterable[Any]]) -> pd.DataFrame:
    """Kline satırlarını (REST dizileri veya CSV satırları) ham çerçeveye çevirir.

    Args:
        rows: 12 alanlı kline satırları.

    Returns:
        ``open_time`` (UTC) indeksli, sayısal OHLCV çerçevesi. Boş girdide boş
        çerçeve döner.
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    # Sütun sayısı sürümler arasında değişebilir; ilk 12'yi konumsal alırız.
    df = df.iloc[:, : len(KLINE_COLUMNS)]
    df.columns = list(KLINE_COLUMNS[: df.shape[1]])

    # Başlık satırı varsa open_time sayıya çevrilemez -> düşer.
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[df["open_time"].notna()]
    if df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    unit = infer_epoch_unit(df["open_time"].iloc[0])
    idx = pd.to_datetime(df["open_time"].astype("int64"), unit=unit, utc=True)

    out = pd.DataFrame(index=idx)
    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(df[col].to_numpy(), errors="coerce")
    out.index.name = "timestamp"
    return out.dropna(subset=["open", "high", "low", "close"])


def parse_kline_zip(payload: bytes) -> pd.DataFrame:
    """data.binance.vision zip'inden kline çerçevesi çıkarır.

    Args:
        payload: Zip dosyasının ham baytları.

    Returns:
        Ham OHLCV çerçevesi.

    Raises:
        ValueError: Zip içinde CSV yoksa.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("Zip içinde CSV bulunamadı.")
        with zf.open(names[0]) as fh:
            raw = pd.read_csv(fh, header=None, dtype=str)
    return klines_to_frame(raw.to_numpy().tolist())


def normalize(
    df: pd.DataFrame,
    bar_minutes: int = 1,
    drop_unclosed: bool = True,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Ham çerçeveyi hattın OHLCV sözleşmesine sokar (README bölüm 10).

    Uygulananlar: artan sıraya alma, tekrar eden zaman damgalarını atma,
    :func:`data.validate_ohlcv` doğrulaması ve **kapanmamış son barın
    atılması**.

    Args:
        df: Ham OHLCV çerçevesi.
        bar_minutes: Bar periyodu.
        drop_unclosed: Kapanmamış son barı at.
        now: "Şimdi" (test için).

    Returns:
        Sözleşmeye uygun bar tablosu.
    """
    if df.empty:
        idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=list(OHLCV_COLUMNS), index=idx, dtype="float64")

    out = df.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = validate_ohlcv(out)
    if drop_unclosed:
        out = drop_unclosed_bar(out, now=now, bar_minutes=bar_minutes)
    return out


# --------------------------------------------------------------------------- #
# Kaynak 1: toplu döküm (data.binance.vision)
# --------------------------------------------------------------------------- #


def bulk_urls(symbol: str, start: datetime, end: datetime) -> list[str]:
    """İstenen aralığı kapsayan toplu döküm URL'lerini üretir.

    Tam aylar için aylık zip, kalan günler için günlük zip kullanılır: aylık
    dökümler yalnızca TAMAMLANMIŞ aylar için yayımlanır.

    Args:
        symbol: Binance sembolü.
        start: Başlangıç (UTC).
        end: Bitiş (UTC).

    Returns:
        Denenecek URL listesi (kronolojik).
    """
    urls: list[str] = []
    # Bitişin ait olduğu ayın başı: bundan öncesi "tamamlanmış ay" sayılır.
    current_month_start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while month < current_month_start:
        urls.append(
            f"{BULK_BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{month:%Y-%m}.zip"
        )
        month = (month + timedelta(days=32)).replace(day=1)

    day = max(current_month_start, start.replace(hour=0, minute=0, second=0, microsecond=0))
    while day <= end:
        urls.append(f"{BULK_BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{day:%Y-%m-%d}.zip")
        day += timedelta(days=1)
    return urls


def download_bulk(
    symbol: str,
    start: datetime,
    end: datetime,
    getter: Callable[[str], bytes] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Toplu dökümlerden veri indirir.

    Eksik (404) dosyalar atlanır: aylık döküm henüz yayımlanmamış veya sembol o
    dönemde işlem görmemiş olabilir. Bu bir hata değil, ölçülecek bir olgudur.

    Args:
        symbol: Binance sembolü.
        start: Başlangıç (UTC).
        end: Bitiş (UTC).
        getter: Test için enjekte edilebilen indirici.

    Returns:
        ``(ham_cerceve, meta)``. ``meta`` indirilen/eksik dosya sayısını taşır.
    """
    fetch = getter or (lambda u: http_get_retry(u))
    urls = bulk_urls(symbol, start, end)
    frames: list[pd.DataFrame] = []
    ok = missing = failed = 0

    for u in urls:
        name = u.rsplit("/", 1)[-1]
        try:
            payload = fetch(u)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                missing += 1
                continue
            failed += 1
            _log(f"  HATA {e.code}: {name}")
            continue
        except Exception as e:  # noqa: BLE001
            failed += 1
            _log(f"  HATA: {name} ({type(e).__name__})")
            continue
        try:
            frames.append(parse_kline_zip(payload))
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            _log(f"  ayristirilamadi: {name} ({e})")

    meta = {"kaynak": "bulk", "dosya_ok": ok, "dosya_404": missing, "dosya_hata": failed,
            "dosya_toplam": len(urls)}
    if not frames:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS)), meta
    return pd.concat(frames), meta


# --------------------------------------------------------------------------- #
# Kaynak 2: REST sayfalama
# --------------------------------------------------------------------------- #


def download_rest(
    symbol: str,
    start: datetime,
    end: datetime,
    limit: int = 1000,
    pause: float = 0.25,
    getter: Callable[[str], bytes] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """REST ucundan ileri sayfalama ile veri indirir.

    Args:
        symbol: Binance sembolü.
        start: Başlangıç (UTC).
        end: Bitiş (UTC).
        limit: Sayfa başına bar (Binance üst sınırı 1000).
        pause: İstekler arası bekleme (rate-limit nezaketi).
        getter: Test için enjekte edilebilen indirici.

    Returns:
        ``(ham_cerceve, meta)``.
    """
    fetch = getter or (lambda u: http_get_retry(u))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    pages = 0

    while start_ms < end_ms:
        url = (
            f"{REST_URL}?symbol={symbol}&interval=1m&limit={limit}"
            f"&startTime={start_ms}&endTime={end_ms}"
        )
        try:
            rows = json.loads(fetch(url).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            _log(f"  REST hatasi ({type(e).__name__}); sayfalama durduruldu.")
            break
        if not rows:
            break
        frame = klines_to_frame(rows)
        if frame.empty:
            break
        frames.append(frame)
        pages += 1
        # Bir sonraki sayfa: son barın açılışından bir bar sonrası.
        last_open_ms = int(frame.index[-1].timestamp() * 1000)
        nxt = last_open_ms + 60_000
        if nxt <= start_ms:  # ilerleme yoksa sonsuz döngüyü kes
            break
        start_ms = nxt
        if pages % 10 == 0:
            _log(f"  {pages} sayfa, {sum(len(f) for f in frames):,} bar...")
        time.sleep(pause)

    meta = {"kaynak": "rest", "sayfa": pages}
    if not frames:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS)), meta
    return pd.concat(frames), meta


# --------------------------------------------------------------------------- #
# Sembol seçimi: "ölç, varsayma"
# --------------------------------------------------------------------------- #


def probe_symbol(
    symbol: str,
    days: int = 2,
    downloader: Callable[[str, datetime, datetime], tuple[pd.DataFrame, dict]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bir sembolün gerçekten veri döndürüp döndürmediğini ve dolduğunu ÖLÇER.

    "BTCTRY vardır" varsaymak yerine son birkaç günü indirip **doluluk oranını**
    (gerçek bar / beklenen bar) hesaplarız. İnce (thin) pariteler saatlerce
    işlem görmez; böyle bir seride eğitilen model gerçekte olmayan boşlukları
    öğrenir.

    Args:
        symbol: Denenecek sembol.
        days: Kaç günlük örnek alınacağı.
        downloader: İndirici (test için enjekte edilebilir).
        now: "Şimdi" (test için).

    Returns:
        ``symbol``, ``bar``, ``beklenen``, ``doluluk``, ``kullanilabilir``,
        ``hata`` alanlı ölçüm sözlüğü.
    """
    dl = downloader or (lambda s, a, b: download_rest(s, a, b))
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        raw, _ = dl(symbol, start, end)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "bar": 0, "beklenen": days * 1440,
                "doluluk": 0.0, "kullanilabilir": False, "hata": f"{type(e).__name__}: {e}"}

    bars = normalize(raw, now=pd.Timestamp(end))
    expected = max(days * 1440, 1)
    coverage = len(bars) / expected
    return {
        "symbol": symbol,
        "bar": int(len(bars)),
        "beklenen": expected,
        "doluluk": float(coverage),
        "kullanilabilir": bool(len(bars) > 0 and coverage >= MIN_COVERAGE),
        "hata": "",
    }


def choose_symbol(
    candidates: Iterable[str] = DEFAULT_SYMBOLS,
    days: int = 2,
    downloader: Callable[[str, datetime, datetime], tuple[pd.DataFrame, dict]] | None = None,
    now: datetime | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Adayları sırayla ölçer ve ilk KULLANILABİLİR olanı seçer.

    Tercih sırası korunur: TL paritesi Paribu'ya daha yakın olduğu için önce
    denenir; yeterince dolu değilse USDT paritesine düşülür.

    Args:
        candidates: Aday semboller (tercih sırasıyla).
        days: Ölçüm penceresi.
        downloader: İndirici.
        now: "Şimdi".

    Returns:
        ``(secilen_sembol_veya_None, tum_olcumler)``.
    """
    reports: list[dict[str, Any]] = []
    chosen: str | None = None
    for sym in candidates:
        rep = probe_symbol(sym, days=days, downloader=downloader, now=now)
        reports.append(rep)
        status = "kullanilabilir" if rep["kullanilabilir"] else "yetersiz"
        detail = rep["hata"] or f"{rep['bar']:,}/{rep['beklenen']:,} bar (doluluk {rep['doluluk']:.1%})"
        _log(f"  {sym}: {status} — {detail}")
        if rep["kullanilabilir"] and chosen is None:
            chosen = sym
            break  # tercih sırası: ilk uygun olanı al
    return chosen, reports


# --------------------------------------------------------------------------- #
# Üst seviye akış
# --------------------------------------------------------------------------- #


def parse_date(value: str | datetime | None) -> datetime | None:
    """``"2022-05-01"`` gibi bir tarihi UTC ``datetime``'a çevirir.

    Args:
        value: Tarih dizgesi veya ``datetime``.

    Returns:
        UTC ``datetime`` veya ``None``.

    Raises:
        ValueError: Biçim tanınmıyorsa.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(str(value))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def fetch_history(
    symbol: str | None = None,
    days: int = 60,
    source: str = "auto",
    out_dir: str | Path | None = None,
    now: datetime | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    tag: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Geçmiş barları indirir, sözleşmeye sokar ve parquet'e yazar.

    Args:
        symbol: Sembol. ``None`` ise :data:`DEFAULT_SYMBOLS` ölçülerek seçilir.
        days: Kaç günlük geçmiş (``start``/``end`` verilmediyse kullanılır).
        source: ``"auto"`` (önce toplu, olmazsa REST), ``"bulk"`` veya ``"rest"``.
        out_dir: Çıktı klasörü. ``None`` ise ``CollectorConfig.ohlcv_dir``.
        now: "Şimdi" (test için).
        start: Açık başlangıç tarihi (örn. ``"2022-05-01"``). Verilirse
            ``days`` yok sayılır.
        end: Açık bitiş tarihi. ``start`` verilip ``end`` verilmezse
            ``start + days`` kullanılır.
        tag: Dosya adına eklenecek etiket (örn. ``"luna"``). Farklı dönemlerin
            birbirinin üstüne yazmasını engeller.

    Returns:
        ``(bars, meta)``. ``meta`` seçilen sembolü, kaynağı ve ölçümleri taşır.

    Notes:
        Tarihsel bir pencere indirirken sembol ÖLÇÜMÜ de o pencereden yapılır;
        aksi hâlde "bugün BTCTRY var" diye 2021'e bakılırdı — oysa parite o
        tarihte listelenmemiş olabilir.
    """
    explicit_start = parse_date(start)
    explicit_end = parse_date(end)
    if explicit_start is not None:
        start_dt = explicit_start
        end_dt = explicit_end or (start_dt + timedelta(days=days))
    elif explicit_end is not None:
        end_dt = explicit_end
        start_dt = end_dt - timedelta(days=days)
    else:
        end_dt = now or datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
    end, start = end_dt, start_dt

    def bulk_dl(s: str, a: datetime, b: datetime) -> tuple[pd.DataFrame, dict]:
        return download_bulk(s, a, b)

    def rest_dl(s: str, a: datetime, b: datetime) -> tuple[pd.DataFrame, dict]:
        return download_rest(s, a, b)

    primary = bulk_dl if source in ("auto", "bulk") else rest_dl

    # --- 1) Sembol seçimi: ölç, varsayma --------------------------------- #
    reports: list[dict[str, Any]] = []
    if symbol is None:
        _log("Sembol olculuyor (once BTCTRY, yetersizse BTCUSDT)...")
        # Ölçüm REST ile daha ucuz; toplu döküm bir aylık zip indirmeyi gerektirir.
        probe_dl = rest_dl if source != "bulk" else bulk_dl
        # Ölçüm penceresi istenen dönemin SONU olmalı: sembolün bugün var olması,
        # 2021'de de listelendiği anlamına gelmez.
        symbol, reports = choose_symbol(DEFAULT_SYMBOLS, days=2, downloader=probe_dl, now=end)
        if symbol is None:
            _log("Hicbir aday sembol kullanilabilir veri dondurmedi.")
            return pd.DataFrame(), {"secilen_sembol": None, "olcumler": reports}
        _log(f"SECILEN SEMBOL: {symbol}")

    # --- 2) İndirme -------------------------------------------------------- #
    _log(f"{symbol} icin ~{days} gunluk 1m bar indiriliyor (kaynak={source})...")
    raw, meta = primary(symbol, start, end)

    if raw.empty and source == "auto":
        _log("Toplu dokum bos dondu; REST'e dusuluyor.")
        raw, meta = rest_dl(symbol, start, end)

    bars = normalize(raw, bar_minutes=1, drop_unclosed=True, now=pd.Timestamp(end))
    bars = bars.loc[bars.index >= pd.Timestamp(start)]

    meta.update({
        "secilen_sembol": symbol,
        "olcumler": reports,
        "istenen_gun": days,
        "bar": int(len(bars)),
    })

    # --- 3) Yazma ---------------------------------------------------------- #
    if not bars.empty:
        d = Path(out_dir) if out_dir else Path(CONFIG.collector.ohlcv_dir)
        d.mkdir(parents=True, exist_ok=True)
        # Dosya adı HANGİ sembolün kullanıldığını açıkça taşır.
        suffix = f"_{tag}" if tag else ""
        if explicit_start is not None or explicit_end is not None:
            suffix += f"_{start:%Y%m%d}_{end:%Y%m%d}"
        path = d / f"{symbol}_1m_binance{suffix}.parquet"
        bars.to_parquet(path, index=True)
        meta["dosya"] = str(path)
    return bars, meta


def summarize(bars: pd.DataFrame, meta: dict[str, Any]) -> None:
    """İndirme sonucunu ekrana basar (fiyatlar gözle doğrulanabilsin diye)."""
    print("\n" + "=" * 74)
    print("INDIRME OZETI")
    print("=" * 74)
    print(f"  Sembol       : {meta.get('secilen_sembol')}")
    print(f"  Kaynak       : {meta.get('kaynak')}")
    if "dosya_toplam" in meta:
        print(f"  Dosya        : {meta['dosya_ok']} ok / {meta['dosya_404']} yok / "
              f"{meta['dosya_hata']} hata (toplam {meta['dosya_toplam']})")
    if "sayfa" in meta:
        print(f"  REST sayfa   : {meta['sayfa']}")
    print(f"  Bar sayisi   : {len(bars):,}")

    if bars.empty:
        print("\n  UYARI: hic bar inmedi. Ag erisimi ve sembol adini kontrol edin.")
        return

    span = bars.index[-1] - bars.index[0]
    expected = int(span.total_seconds() // 60) + 1
    print(f"  Aralik       : {bars.index[0]}  ->  {bars.index[-1]}  ({span.days} gun)")
    print(f"  Doluluk      : {len(bars) / max(expected, 1):.1%} ({len(bars):,}/{expected:,})")
    if meta.get("dosya"):
        print(f"  Yazildi      : {meta['dosya']}")

    print("\n--- Ilk 3 bar ---")
    print(bars.head(3).to_string())
    print("\n--- Son 3 bar ---")
    print(bars.tail(3).to_string())
    print(
        f"\n  Fiyat araligi: {bars['close'].min():,.2f} - {bars['close'].max():,.2f}"
        f"  |  ortalama hacim/bar: {bars['volume'].mean():,.3f}"
    )
    print(
        "\n  HATIRLATMA: Bu veri Binance'tendir, Paribu DEGILDIR. Prototip amaclidir;\n"
        "  canliya gecmeden once Paribu'nun kendi verisiyle yeniden dogrulanmalidir."
    )


def main(argv: list[str] | None = None) -> int:
    """Komut satırı giriş noktası.

    Args:
        argv: Argümanlar (test için).

    Returns:
        Çıkış kodu (veri inmezse 1).
    """
    ap = argparse.ArgumentParser(
        description="Binance public 1m gecmis bar indirici (anahtarsiz, PROTOTIP veri).",
    )
    ap.add_argument("--symbol", type=str, default=None, help="Sembol. Bos ise olculerek secilir.")
    ap.add_argument("--days", type=int, default=60, help="Kac gunluk gecmis (varsayilan 60).")
    ap.add_argument("--source", choices=["auto", "bulk", "rest"], default="auto")
    ap.add_argument("--out-dir", type=str, default=None, help="Cikti klasoru.")
    ap.add_argument("--start", type=str, default=None, help="Baslangic tarihi (2022-05-01).")
    ap.add_argument("--end", type=str, default=None, help="Bitis tarihi (2022-06-15).")
    ap.add_argument("--tag", type=str, default=None, help="Dosya adina eklenecek etiket.")
    ap.add_argument("--probe-only", action="store_true", help="Sadece sembolleri olc, indirme.")
    args = ap.parse_args(argv)

    if args.probe_only:
        _log("Sembol olcumu (indirme yapilmayacak)...")
        chosen, reports = choose_symbol(DEFAULT_SYMBOLS, days=2)
        print("\n--- Olcumler ---")
        for r in reports:
            print(f"  {r['symbol']:10s} bar={r['bar']:>7,}  doluluk={r['doluluk']:.1%}"
                  f"  kullanilabilir={r['kullanilabilir']}  {r['hata']}")
        print(f"\nSecilen: {chosen}")
        return 0 if chosen else 1

    bars, meta = fetch_history(
        symbol=args.symbol, days=args.days, source=args.source, out_dir=args.out_dir,
        start=args.start, end=args.end, tag=args.tag
    )
    summarize(bars, meta)
    return 0 if not bars.empty else 1


if __name__ == "__main__":
    sys.exit(main())
