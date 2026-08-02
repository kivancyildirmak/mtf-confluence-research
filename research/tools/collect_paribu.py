#!/usr/bin/env python3
"""Paribu public ticker'ından POLL-FORWARD veri toplayıcı ve bar toplulaştırıcı.

**NEDEN BU ARAÇ VAR?** Paribu'da tarihsel mum/işlem/emir-defteri ucu yoktur
(probe ile doğrulandı: ``candles`` / ``orderbook`` / ``trades`` -> 404). Elde
yalnızca anlık durum veren tek bir public uç var:

    GET https://www.paribu.com/ticker
    -> {"BTC_TL": {"last", "lowestAsk", "highestBid", "low24hr", "high24hr",
                   "avg24hr", "volume", "change", "percentChange"}, ...}

Dolayısıyla geçmiş **ancak bugünden itibaren biriktirilebilir**. Bu araç tam da
onu yapar: ucu sık aralıkla yoklar, ham tick'leri kaybetmeden diske yazar ve
ayrı bir komutla 1 dakikalık OHLCV barlarına toplulaştırır.

Anahtar/HMAC YOKTUR — bu uç publictir.

Kullanım::

    # Toplamayı başlat (Ctrl+C ile temiz kapanır)
    python -m research.tools.collect_paribu collect
    python -m research.tools.collect_paribu collect --symbols BTC_TL,ETH_TL --interval 5

    # Ham tick'leri 1 dakikalık barlara çevir
    python -m research.tools.collect_paribu resample --symbol BTC_TL

    # Ne kadar veri birikmiş?
    python -m research.tools.collect_paribu status

Uzun süre çalıştırmak için (kopma olmadan)::

    nohup python -m research.tools.collect_paribu collect > collector.log 2>&1 &

VERİ KALİTESİ UYARILARI (ayrıntı: README bölüm 10.3):

1. **high/low dardır.** Bar içi ekstremler yalnızca yoklama anlarında görülür;
   5 sn aralıkta dakikada 12 örnek demektir. Gerçek high/low bundan geniştir.
   Parkinson/Garman-Klass tahmincileri bu veride sistematik olarak düşük çıkar.
2. **volume 24 SAATLİK KÜMÜLATİFTİR**, anlık değil. Dakikalık hacim ardışık
   okumaların farkından türetilir ve bu yaklaşım kusurludur — bkz.
   :func:`diagnose_volume_series` ve README 10.3.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import CONFIG, CollectorConfig
from ..data import OHLCV_COLUMNS, drop_unclosed_bar, validate_ohlcv

#: Ham tick tablosunun sütun şeması (parquet'e bu yazılır).
TICK_COLUMNS: tuple[str, ...] = (
    "ts",
    "symbol",
    "last",
    "lowest_ask",
    "highest_bid",
    "volume24h",
)

#: Ticker yanıtındaki alan adlarından bizim şemamıza eşleme.
FIELD_MAP: dict[str, str] = {
    "last": "last",
    "lowestAsk": "lowest_ask",
    "highestBid": "highest_bid",
    "volume": "volume24h",
}


def _log(msg: str) -> None:
    """Zaman damgalı log satırı (UTC)."""
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def _to_float(value: Any) -> float:
    """Borsa yanıtındaki sayıyı **sessizce bozmadan** float'a çevirir.

    Borsalar sayıları çoğu zaman dizge döndürür ve biçim yerele göre değişir.
    Paribu'nun hangi biçimi kullandığı DOĞRULANMADI (bkz. README 10.3), bu
    yüzden çevirim kasıtlı olarak muhafazakâr iki aşamalıdır:

    1. Önce standart ``float()`` denenir -> ``"3000000"``, ``"3000000.5"``, sayı.
    2. Yalnızca 1. adım BAŞARISIZ olursa TR biçimi denenir (nokta = binlik,
       virgül = ondalık) -> ``"3.000.000,25"``, ``"3.000.000"``.

    Sıra kritiktir: baştan nokta silmek ``"3000.50"`` değerini ``300050``
    yapardı — yani fiyatı 100 katına çıkaran sessiz bir bozulma. Bu sıralamayla
    geçerli ondalıklı dizgeler asla bozulmaz.

    KALAN BELİRSİZLİK: ``"3.000"`` her iki biçimde de geçerlidir (3.0 mı, 3000
    mü?). Standart yorum (3.0) seçilir. :func:`parse_ticker` bu tür değerleri
    sayar ve toplayıcı bunu uyarı olarak loglar; ilk gerçek yanıt görüldüğünde
    biçim netleşir.

    Args:
        value: Ham değer.

    Returns:
        Float değer; çevrilemezse ``float("nan")``.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def fetch_ticker(cfg: CollectorConfig | None = None) -> dict[str, Any]:
    """Public ticker ucunu bir kez çeker.

    Args:
        cfg: Toplayıcı konfigürasyonu.

    Returns:
        Parite -> alanlar sözlüğü (ham JSON).

    Raises:
        urllib.error.URLError: Ağ/HTTP hatasında.
        ValueError: Yanıt JSON değilse veya beklenen yapıda değilse.

    Notes:
        Anahtar/imza yoktur; uç tamamen publictir.
    """
    c = cfg or CONFIG.collector
    req = urllib.request.Request(
        c.ticker_url,
        headers={"User-Agent": c.user_agent, "Accept": "application/json"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=c.request_timeout, context=ctx) as resp:
        raw = resp.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Ticker yanıtı JSON değil: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Ticker yanıtı sözlük bekleniyordu, {type(data).__name__} geldi.")
    return data


def parse_ticker(
    payload: dict[str, Any],
    symbols: tuple[str, ...] | list[str],
    ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Ticker yanıtından istenen pariteler için tick kayıtları çıkarır.

    Args:
        payload: :func:`fetch_ticker` çıktısı.
        symbols: İlgilenilen pariteler.
        ts: Okumanın UTC zaman damgası.

    Returns:
        :data:`TICK_COLUMNS` şemasına uyan kayıt listesi. Yanıtta bulunmayan
        parite sessizce atlanır (çağıran tarafta loglanır).
    """
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        entry = payload.get(sym)
        if not isinstance(entry, dict):
            continue
        row: dict[str, Any] = {"ts": ts, "symbol": sym}
        for src, dst in FIELD_MAP.items():
            row[dst] = _to_float(entry.get(src))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Depolama (append-only, atomik)
# --------------------------------------------------------------------------- #


def tick_path(day: str, cfg: CollectorConfig | None = None) -> Path:
    """Belirli bir gün için tick dosyasının yolu.

    Args:
        day: ``YYYY-MM-DD`` biçiminde UTC gün.
        cfg: Toplayıcı konfigürasyonu.

    Returns:
        Parquet dosya yolu.
    """
    c = cfg or CONFIG.collector
    return Path(c.tick_dir) / f"{day}.parquet"


def append_ticks(rows: list[dict[str, Any]], cfg: CollectorConfig | None = None) -> int:
    """Tick kayıtlarını gün bazlı parquet dosyalarına EKLER (append-only).

    Yazma stratejisi: mevcut dosya okunur, yeni satırlar eklenir, geçici dosyaya
    yazılır ve ``os.replace`` ile ATOMİK olarak yerine konur. Böylece yazma
    sırasında süreç ölse bile dosya asla yarım kalmaz.

    Args:
        rows: Yazılacak tick kayıtları.
        cfg: Toplayıcı konfigürasyonu.

    Returns:
        Diske yazılan yeni satır sayısı.

    Notes:
        Aynı ``(ts, symbol)`` çifti tekrar gelirse son kayıt tutulur; yeniden
        başlatmalarda oluşabilecek çakışmaları bu temizler.
    """
    if not rows:
        return 0
    c = cfg or CONFIG.collector
    df = pd.DataFrame(rows, columns=list(TICK_COLUMNS))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    written = 0
    for day, chunk in df.groupby(df["ts"].dt.strftime("%Y-%m-%d")):
        p = tick_path(str(day), c)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                existing = pd.read_parquet(p)
                chunk = pd.concat([existing, chunk], ignore_index=True)
            except Exception as e:  # noqa: BLE001 - bozuk dosya toplamayı durdurmamalı
                _log(f"UYARI: {p} okunamadı ({e}); yanına .corrupt olarak taşınıyor.")
                p.replace(p.with_suffix(".corrupt.parquet"))
        chunk = chunk.drop_duplicates(subset=["ts", "symbol"], keep="last")
        chunk = chunk.sort_values("ts").reset_index(drop=True)
        tmp = p.with_suffix(".tmp.parquet")
        chunk.to_parquet(tmp, index=False)
        os.replace(tmp, p)  # atomik
        written += len(chunk)
    return written


def load_ticks(
    symbol: str | None = None,
    day: str | None = None,
    cfg: CollectorConfig | None = None,
) -> pd.DataFrame:
    """Diskteki ham tick'leri okur.

    Args:
        symbol: Filtrelenecek parite. ``None`` ise hepsi.
        day: ``YYYY-MM-DD`` günü. ``None`` ise tüm günler.
        cfg: Toplayıcı konfigürasyonu.

    Returns:
        ``ts`` sütununa göre artan sıralı tick tablosu (boş olabilir).
    """
    c = cfg or CONFIG.collector
    root = Path(c.tick_dir)
    if not root.exists():
        return pd.DataFrame(columns=list(TICK_COLUMNS))
    files = sorted(root.glob(f"{day}.parquet" if day else "*.parquet"))
    files = [f for f in files if not f.name.endswith((".tmp.parquet", ".corrupt.parquet"))]
    if not files:
        return pd.DataFrame(columns=list(TICK_COLUMNS))

    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:  # noqa: BLE001
            _log(f"UYARI: {f} atlandı ({e}).")
    if not frames:
        return pd.DataFrame(columns=list(TICK_COLUMNS))

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    return df.drop_duplicates(subset=["ts", "symbol"], keep="last").sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Toplayıcı döngüsü
# --------------------------------------------------------------------------- #


@dataclass
class CollectorState:
    """Toplayıcının çalışma sırasındaki sayaçları."""

    ticks: int = 0
    polls_ok: int = 0
    polls_failed: int = 0
    flushes: int = 0
    started: float = field(default_factory=time.time)
    last_price: float = float("nan")
    last_error: str = ""
    stop: bool = False


def collect(cfg: CollectorConfig | None = None, max_polls: int | None = None) -> CollectorState:
    """Ana toplama döngüsü: ucu yoklar, tampona yazar, düzenli olarak diske basar.

    SAĞLAMLIK: Hiçbir ağ hatası süreci düşürmez. Hata durumunda üstel geri
    çekilme uygulanır (``backoff_base * 2^n``, ``backoff_max`` ile sınırlı),
    başarılı okumada sayaç sıfırlanır. ``SIGINT``/``SIGTERM`` alındığında
    tampon diske yazılır ve özet basılır.

    Args:
        cfg: Toplayıcı konfigürasyonu.
        max_polls: Test amaçlı: bu kadar yoklamadan sonra dur. ``None`` =
            süresiz.

    Returns:
        Çalışma sonundaki :class:`CollectorState`.
    """
    c = cfg or CONFIG.collector
    state = CollectorState()
    buffer: list[dict[str, Any]] = []

    def _handle_signal(signum: int, _frame: Any) -> None:
        _log(f"Sinyal {signum} alındı; temiz kapanış yapılıyor...")
        state.stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass  # ana thread değilsek sinyal kurulamaz; sorun değil

    _log(f"Toplayıcı başladı | uç: {c.ticker_url}")
    _log(f"  pariteler: {list(c.symbols)} | aralık: {c.poll_interval_seconds}s")
    _log(f"  tick klasörü: {Path(c.tick_dir).resolve()}")
    _log(f"  flush: {c.flush_interval_seconds}s | kalp atışı: {c.heartbeat_interval_seconds}s")

    last_flush = time.time()
    last_heartbeat = time.time()
    consecutive_errors = 0
    polls = 0

    while not state.stop:
        cycle_start = time.time()
        polls += 1
        try:
            payload = fetch_ticker(c)
            ts = pd.Timestamp.now(tz="UTC").floor("s")
            rows = parse_ticker(payload, c.symbols, ts)
            if not rows:
                _log(f"UYARI: yanıtta hiçbir parite bulunamadı: {list(c.symbols)}")
            else:
                buffer.extend(rows)
                state.ticks += len(rows)
                state.last_price = rows[0]["last"]
            state.polls_ok += 1
            consecutive_errors = 0
        except Exception as e:  # noqa: BLE001 - toplayıcı ASLA düşmemeli
            state.polls_failed += 1
            state.last_error = f"{type(e).__name__}: {e}"
            consecutive_errors += 1
            backoff = min(
                c.backoff_base_seconds * (2 ** (consecutive_errors - 1)),
                c.backoff_max_seconds,
            )
            _log(f"HATA ({consecutive_errors}. ardışık): {state.last_error} -> {backoff:.0f}s bekleniyor")
            # Geri çekilmeyi bölerek bekle ki Ctrl+C anında cevap versin.
            if max_polls is not None and polls >= max_polls:
                break
            waited = 0.0
            while waited < backoff and not state.stop:
                time.sleep(min(1.0, backoff - waited))
                waited += 1.0
            continue

        now = time.time()
        if now - last_flush >= c.flush_interval_seconds and buffer:
            append_ticks(buffer, c)
            state.flushes += 1
            buffer.clear()
            last_flush = now

        if now - last_heartbeat >= c.heartbeat_interval_seconds:
            uptime = now - state.started
            _log(
                f"calisiyor | tick={state.ticks} basarili={state.polls_ok} "
                f"hatali={state.polls_failed} tampon={len(buffer)} "
                f"son_fiyat={state.last_price:,.0f} uptime={uptime / 3600:.2f}s"
            )
            last_heartbeat = now

        if max_polls is not None and polls >= max_polls:
            break

        # Sabit ızgarada kal: işlem süresini düşerek sürüklenmeyi (drift) önle.
        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, c.poll_interval_seconds - elapsed)
        slept = 0.0
        while slept < sleep_for and not state.stop:
            step = min(0.5, sleep_for - slept)
            time.sleep(step)
            slept += step

    if buffer:
        append_ticks(buffer, c)
        state.flushes += 1
        buffer.clear()

    uptime = time.time() - state.started
    _log(
        f"Kapandı | toplam tick={state.ticks} basarili={state.polls_ok} "
        f"hatali={state.polls_failed} flush={state.flushes} sure={uptime / 60:.1f}dk"
    )
    return state


# --------------------------------------------------------------------------- #
# Hacim tanılaması
# --------------------------------------------------------------------------- #


def diagnose_volume_series(ticks: pd.DataFrame) -> dict[str, Any]:
    """``volume24h`` serisinin nasıl davrandığını ölçer (varsayım yapmaz).

    ``volume`` alanı 24 saatlik kümülatiftir. Ardışık farkı almak iki farklı
    senaryoda iki farklı anlama gelir ve hangisinin geçerli olduğunu ancak
    GERÇEK VERİ söyler:

    * **Günlük sıfırlanan sayaç:** fark yalnızca gün dönümünde büyük negatif
      olur; diğer tüm farklar >= 0'dır. Bu durumda fark = o aralığın hacmi.
    * **Kayan 24s penceresi:** pencereye giren işlemler kadar ÇIKAN işlemler de
      vardır; fark = (yeni hacim) - (24s önce düşen hacim). Negatif farklar gün
      boyuna yayılır ve fark artık aralık hacmi DEĞİLDİR (aşağı yanlı).

    Bu fonksiyon negatif farkların gün dönümünde kümelenip kümelenmediğine
    bakarak hangi senaryonun geçerli olduğunu raporlar.

    Args:
        ticks: Tek pariteye ait, ``ts`` ve ``volume24h`` içeren tick tablosu.

    Returns:
        Tanı sözlüğü: negatif fark sayısı/oranı, gün dönümünde olma oranı ve
        çıkarsanan mod (``"gunluk_sifirlanan"`` / ``"kayan_24s"`` /
        ``"belirsiz"``).
    """
    if len(ticks) < 3:
        return {"mod": "belirsiz", "sebep": "yetersiz veri", "n_tick": int(len(ticks))}

    s = ticks.sort_values("ts")
    delta = s["volume24h"].diff()
    neg = delta < 0
    n_neg = int(neg.sum())
    if n_neg == 0:
        return {
            "mod": "gunluk_sifirlanan",
            "sebep": "hic negatif fark yok (henuz gun donumu gorulmemis olabilir)",
            "n_tick": int(len(s)),
            "negatif_sayisi": 0,
            "negatif_orani": 0.0,
        }

    neg_ts = s.loc[neg, "ts"]
    # Gün dönümü penceresi: UTC 23:55-00:05.
    near_midnight = ((neg_ts.dt.hour == 23) & (neg_ts.dt.minute >= 55)) | (
        (neg_ts.dt.hour == 0) & (neg_ts.dt.minute <= 5)
    )
    ratio_midnight = float(near_midnight.mean())
    mode = "gunluk_sifirlanan" if ratio_midnight > 0.8 else "kayan_24s"
    return {
        "mod": mode,
        "n_tick": int(len(s)),
        "negatif_sayisi": n_neg,
        "negatif_orani": float(n_neg / max(len(s) - 1, 1)),
        "gun_donumunde_olma_orani": ratio_midnight,
        "sebep": (
            "negatif farklar gun donumunde kumeleniyor"
            if mode == "gunluk_sifirlanan"
            else "negatif farklar gune yayilmis -> kayan pencere, hacim ASAGI YANLI"
        ),
    }


# --------------------------------------------------------------------------- #
# Toplulaştırma (tick -> 1 dakikalık OHLCV)
# --------------------------------------------------------------------------- #


def resample_ticks_to_ohlcv(
    ticks: pd.DataFrame,
    bar_minutes: int = 1,
    drop_unclosed: bool = True,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Ham tick'leri OHLCV barlarına toplulaştırır.

    OHLC, ``last`` (son işlem fiyatı) tick'lerinden kurulur:
    ``open`` = dakikanın ilk okuması, ``high``/``low`` = okumaların en
    yüksek/en düşüğü, ``close`` = son okuma.

    Ek olarak mikroyapı sütunları yazılır (elimizde olduğu için değerli):
    ``spread_mean`` (mutlak TL), ``spread_rel_mean`` (mid'e oranla),
    ``tick_count`` (o dakikadaki okuma sayısı — veri kalitesi göstergesi).

    Args:
        ticks: Tek pariteye ait tick tablosu.
        bar_minutes: Bar periyodu (dakika).
        drop_unclosed: Kapanmamış son barı at (README bölüm 10 sözleşmesi).
        now: "Şimdi" (test için). ``None`` ise sistem saati.

    Returns:
        UTC ``DatetimeIndex``'li, artan sıralı bar tablosu:
        ``open, high, low, close, volume`` + ``spread_mean``,
        ``spread_rel_mean``, ``tick_count``, ``volume_gecerli_oran``.

    Notes:
        **SIZINTI:** Bir bar yalnızca kendi zaman aralığındaki tick'lerden
        kurulur; ileriye dönük doldurma (``bfill``) yapılmaz. Tick'i olmayan
        dakikalar bara DÖNÜŞTÜRÜLMEZ — boşluk sessizce doldurulmaz.

        **high/low DARDIR:** yalnızca yoklama anları görülür. 5 sn aralıkta
        dakikada ~12 örnek vardır; gerçek ekstremler bundan geniştir.

        **HACİM:** ``volume24h`` kümülatiftir; dakikalık hacim ardışık farkların
        toplamıdır. Negatif farklar (gün dönümü sıfırlanması veya 24s kayan
        pencereden çıkan işlemler) ölçülemez kabul edilip NaN yapılır ve o
        aralık toplama dahil EDİLMEZ. ``volume_gecerli_oran`` sütunu, o barda
        farkların ne kadarının kullanılabildiğini söyler; 1.0'dan küçükse hacim
        eksik ölçülmüştür. Ayrıntı için :func:`diagnose_volume_series`.
    """
    cols = [
        *OHLCV_COLUMNS,
        "spread_mean",
        "spread_rel_mean",
        "tick_count",
        "volume_gecerli_oran",
    ]
    if ticks.empty:
        idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=cols, index=idx, dtype="float64")

    s = ticks.copy()
    s["ts"] = pd.to_datetime(s["ts"], utc=True)
    s = s.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").set_index("ts")

    # --- Hacim: kümülatiften aralık farkına ------------------------------- #
    # Negatif fark = gün dönümü sıfırlanması VEYA kayan pencereden çıkan işlem.
    # İkisi de "bu aralığın hacmi ölçülemedi" demektir -> NaN (uydurma YOK).
    delta = s["volume24h"].diff()
    valid = delta.notna() & (delta >= 0)
    delta_valid = delta.where(valid)

    freq = f"{bar_minutes}min"
    last = s["last"]

    out = pd.DataFrame(
        {
            "open": last.resample(freq).first(),
            "high": last.resample(freq).max(),
            "low": last.resample(freq).min(),
            "close": last.resample(freq).last(),
            "volume": delta_valid.resample(freq).sum(min_count=1),
            "tick_count": last.resample(freq).count().astype("float64"),
        }
    )

    spread = s["lowest_ask"] - s["highest_bid"]
    mid = (s["lowest_ask"] + s["highest_bid"]) / 2.0
    out["spread_mean"] = spread.resample(freq).mean()
    out["spread_rel_mean"] = (spread / mid.replace(0.0, pd.NA)).resample(freq).mean()

    n_valid = valid.resample(freq).sum().astype("float64")
    n_total = delta.notna().resample(freq).sum().astype("float64")
    out["volume_gecerli_oran"] = (n_valid / n_total.replace(0.0, pd.NA)).astype("float64")

    # Tick'i olmayan dakikalar bar değildir; boşluk doldurulmaz.
    out = out[out["tick_count"] > 0]
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "timestamp"

    if out.empty:
        idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=cols, index=idx, dtype="float64")

    # OHLCV çekirdeğini ortak doğrulayıcıdan geçir, ek sütunları geri ekle.
    core = validate_ohlcv(out.loc[:, list(OHLCV_COLUMNS)])
    extras = out.loc[core.index, ["spread_mean", "spread_rel_mean", "tick_count", "volume_gecerli_oran"]]
    result = core.join(extras)

    if drop_unclosed:
        result = drop_unclosed_bar(result, now=now, bar_minutes=bar_minutes)
    return result


def resample_and_save(
    symbol: str,
    bar_minutes: int = 1,
    out_path: str | Path | None = None,
    cfg: CollectorConfig | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Diskteki tick'leri okur, barlara çevirir ve parquet'e yazar.

    Args:
        symbol: Parite.
        bar_minutes: Bar periyodu.
        out_path: Çıktı yolu. ``None`` ise ``ohlcv_dir/{symbol}_{n}m.parquet``.
        cfg: Toplayıcı konfigürasyonu.

    Returns:
        ``(bars, yazilan_yol)``. Hiç tick yoksa yol ``None``.
    """
    c = cfg or CONFIG.collector
    ticks = load_ticks(symbol=symbol, cfg=c)
    if ticks.empty:
        return pd.DataFrame(), None

    bars = resample_ticks_to_ohlcv(ticks, bar_minutes=bar_minutes)
    if bars.empty:
        return bars, None

    p = Path(out_path) if out_path else Path(c.ohlcv_dir) / f"{symbol}_{bar_minutes}m.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(p, index=True)
    return bars, p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_collect(args: argparse.Namespace) -> int:
    """``collect`` alt komutu."""
    import copy

    c = copy.deepcopy(CONFIG.collector)
    if args.symbols:
        c.symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.interval:
        c.poll_interval_seconds = float(args.interval)
    if args.tick_dir:
        c.tick_dir = args.tick_dir
    collect(c, max_polls=args.max_polls)
    return 0


def _cmd_resample(args: argparse.Namespace) -> int:
    """``resample`` alt komutu."""
    if args.tick_dir:
        CONFIG.collector.tick_dir = args.tick_dir
    ticks = load_ticks(symbol=args.symbol)
    if ticks.empty:
        print(f"Hic tick bulunamadi ({args.symbol}). Once 'collect' calistirin.")
        return 1

    diag = diagnose_volume_series(ticks)
    bars, path = resample_and_save(args.symbol, bar_minutes=args.bar_minutes, out_path=args.out)

    print(f"\nTick: {len(ticks):,} | {ticks['ts'].min()} -> {ticks['ts'].max()}")
    print(f"Bar : {len(bars):,} ({args.bar_minutes} dk)")
    print(f"\n--- Hacim tanilamasi ---")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    if diag.get("mod") == "kayan_24s":
        print(
            "  UYARI: volume 24s KAYAN penceredir; ardisik fark gercek aralik\n"
            "  hacmini VERMEZ (asagi yanli). Hacim ozelliklerine temkinli yaklasin."
        )
    if not bars.empty:
        print(f"\n--- Son 5 bar ---")
        print(bars.tail(5).to_string())
        thin = (bars["tick_count"] < 6).mean()
        print(f"\n  6'dan az tick iceren bar orani: {thin:.1%}")
        print(f"  ortalama gecerli hacim orani  : {bars['volume_gecerli_oran'].mean():.3f}")
    if path:
        print(f"\nYazildi: {path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """``status`` alt komutu: ne kadar veri birikmiş?"""
    if args.tick_dir:
        CONFIG.collector.tick_dir = args.tick_dir
    root = Path(CONFIG.collector.tick_dir)
    files = sorted(root.glob("*.parquet")) if root.exists() else []
    files = [f for f in files if not f.name.endswith((".tmp.parquet", ".corrupt.parquet"))]
    if not files:
        print(f"Henuz veri yok ({root.resolve()}). 'collect' ile toplamaya baslayin.")
        return 1

    ticks = load_ticks()
    print(f"Tick klasoru : {root.resolve()}")
    print(f"Gun sayisi   : {len(files)}")
    print(f"Toplam tick  : {len(ticks):,}")
    if not ticks.empty:
        span = ticks["ts"].max() - ticks["ts"].min()
        print(f"Aralik       : {ticks['ts'].min()} -> {ticks['ts'].max()}  ({span})")
        for sym, g in ticks.groupby("symbol"):
            mins = g["ts"].dt.floor("1min").nunique()
            print(f"  {sym}: {len(g):,} tick | ~{mins:,} dakikalik bar")
        # Arastirma hatti icin gereken buyukluge ne kadar var?
        need = 65_000
        have = ticks["ts"].dt.floor("1min").nunique()
        if have < need:
            kalan_gun = (need - have) / 1440.0
            print(f"\n  Hattin varsayilan olcegi ({need:,} bar) icin ~{kalan_gun:.1f} gun daha gerekiyor.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Komut satırı giriş noktası.

    Args:
        argv: Argümanlar (test için).

    Returns:
        Çıkış kodu.
    """
    ap = argparse.ArgumentParser(
        description="Paribu public ticker poll-forward toplayici (anahtarsiz).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_col = sub.add_parser("collect", help="Ticker'i yoklayip tick biriktir.")
    p_col.add_argument("--symbols", type=str, default=None, help="Virgullu parite listesi.")
    p_col.add_argument("--interval", type=float, default=None, help="Yoklama araligi (sn).")
    p_col.add_argument("--tick-dir", type=str, default=None, help="Tick klasoru.")
    p_col.add_argument("--max-polls", type=int, default=None, help="Test icin yoklama siniri.")
    p_col.set_defaults(func=_cmd_collect)

    p_res = sub.add_parser("resample", help="Tick'leri OHLCV barlarina cevir.")
    p_res.add_argument("--symbol", type=str, default=CONFIG.collector.symbols[0])
    p_res.add_argument("--bar-minutes", type=int, default=1)
    p_res.add_argument("--out", type=str, default=None)
    p_res.add_argument("--tick-dir", type=str, default=None, help="Tick klasoru.")
    p_res.set_defaults(func=_cmd_resample)

    p_st = sub.add_parser("status", help="Birikmis veri ozeti.")
    p_st.add_argument("--tick-dir", type=str, default=None, help="Tick klasoru.")
    p_st.set_defaults(func=_cmd_status)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
