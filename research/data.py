"""Veri katmanı.

Bu modül hattın DIŞ DÜNYAYA açılan tek kapısıdır. Şu an sadece yerel dosyadan
(CSV/Parquet) okur ve test için sentetik OHLCV üretir. Paribu API bağlantısı
bilinçli olarak stub bırakılmıştır (bkz. :func:`fetch_paribu_ohlcv` ve
:func:`fetch_orderbook`).

Tasarım kuralı: Araştırma hattının geri kalanı (features/labeling/model) veri
kaynağını ASLA bilmez. Her fonksiyon aynı sözleşmeyi döndürür:

    DatetimeIndex (UTC, artan, tekilleştirilmiş) + sütunlar:
    ``open, high, low, close, volume``

Sızıntı (leakage) notu: Bu katmanda tek sızıntı riski, henüz KAPANMAMIŞ barın
veri setine girmesidir. Canlı botta son bar çoğu zaman yarı-oluşmuş gelir;
:func:`drop_unclosed_bar` bu barı atmak için vardır ve canlı tarafta MUTLAKA
çağrılmalıdır.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import CONFIG, DataConfig, ResearchConfig

#: Hattın her yerinde beklenen standart OHLCV sütunları.
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Emir defteri anlık görüntüsü için beklenen sütunlar.
ORDERBOOK_COLUMNS: tuple[str, ...] = (
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "bid_depth",
    "ask_depth",
)


# --------------------------------------------------------------------------- #
# Canlı API stub'ları — TODO: Paribu bağlantısı sonra yapılacak
# --------------------------------------------------------------------------- #


def fetch_paribu_ohlcv(
    symbol: str,
    interval_minutes: int = 1,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Paribu OHLCV barlarını YEREL poll-forward deposundan okur.

    **Paribu'da tarihsel mum ucu YOKTUR** (probe ile doğrulandı: ``candles`` /
    ``orderbook`` / ``trades`` -> 404). Tek public uç anlık ticker'dır
    (``https://www.paribu.com/ticker``). Bu yüzden "geçmiş veri çekmek" mümkün
    değildir; geçmiş ancak :mod:`research.tools.collect_paribu` ile BUGÜNDEN
    İTİBAREN biriktirilir. Bu fonksiyon o birikimi okur ve hattın geri kalanına
    standart OHLCV sözleşmesiyle sunar.

    VERİ KALİTESİ: Barlar ticker anlık görüntülerinden kurulduğu için
    ``high``/``low`` gerçeğinden DARDIR ve ``volume`` 24 saatlik kümülatifin
    farkından türetilir. İkisi de README bölüm 10.3'te ayrıntılı anlatılan
    yanlılıklar taşır; Parkinson/Garman-Klass ve hacim özelliklerini
    yorumlarken bu göz önünde tutulmalıdır.

    Args:
        symbol: Paribu sembolü (örn. ``"BTC_TL"``).
        interval_minutes: Mum periyodu (dakika).
        start: Başlangıç zamanı (UTC). ``None`` ise borsanın verdiği en eski bar.
        end: Bitiş zamanı (UTC). ``None`` ise şimdiki an.
        limit: Maksimum bar sayısı.

    Returns:
        ``open, high, low, close, volume`` sütunlu, UTC indeksli DataFrame.
        Toplayıcının ürettiği ek mikroyapı sütunları (``spread_mean``,
        ``spread_rel_mean``, ``tick_count``, ``volume_gecerli_oran``) varsa
        korunur.

    Raises:
        FileNotFoundError: Henüz hiç veri toplanmamışsa (toplayıcı hiç
            çalıştırılmamış).

    Notes:
        Bu fonksiyon ağa GİTMEZ; yerel poll-forward deposunu okur. Sözleşme
        gereği: UTC indeks, artan sıra, tekilleştirme ve **kapanmamış son barın
        atılması** :func:`tools.collect_paribu.resample_ticks_to_ohlcv`
        tarafından uygulanır.

        Eksik barlar (toplayıcının durduğu dönemler) sessizce doldurulmaz —
        boşluk olarak kalır. Doldurmak isterseniz :func:`align_to_bars`
        açıkça çağrılmalıdır.
    """
    # Yerel (lazy) import: tools.collect_paribu bu modülden import ettiği için
    # modül seviyesinde import edilirse döngüsel bağımlılık oluşur.
    from .tools.collect_paribu import load_ticks, resample_ticks_to_ohlcv

    ticks = load_ticks(symbol=symbol)
    if ticks.empty:
        raise FileNotFoundError(
            f"'{symbol}' için toplanmış tick verisi yok.\n"
            "Paribu'da tarihsel mum ucu bulunmadığı için geçmiş, ancak ileriye\n"
            "dönük toplanabilir. Toplayıcıyı başlatın:\n"
            "    python -m research.tools.collect_paribu collect\n"
            "Ayrıntı: research/README.md bölüm 10.3"
        )

    bars = resample_ticks_to_ohlcv(ticks, bar_minutes=interval_minutes)

    if start is not None:
        bars = bars.loc[bars.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        bars = bars.loc[bars.index <= pd.Timestamp(end, tz="UTC")]
    if limit is not None and len(bars) > limit:
        bars = bars.iloc[-int(limit) :]  # en YENİ barlar tutulur
    return bars


def fetch_orderbook(
    symbol: str,
    depth: int = 20,
    snapshot_interval_minutes: int = 1,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Paribu emir defteri (order book) anlık görüntülerini çeker. **(STUB)**

    DURUM (2026-08-02): :func:`fetch_paribu_ohlcv` ile aynı sebepten
    bağlanamadı — ``paribu.com`` hostları geliştirme ortamının çıkış
    politikasınca 403 ile kapalı, doküman okunamadı. Bkz. README bölüm 10.1.

    AYRICA ÖNEMLİ: Emir defterinin **tarihsel** verisi hiçbir kaynakta yoktur;
    hiçbir borsa geçmiş defter anlık görüntüsü sunmaz. Dolayısıyla ``mk_*``
    özellikleri ancak ileriye dönük toplamayla (README 10.2, seçenek "a")
    doldurulabilir. Üçüncü parti tarihsel bar verisiyle çalışılırsa bu sütunlar
    NaN kalır — hat bunu zaten destekler.

    Mikroyapı özellikleri (spread, imbalance, derinlik) bu veriyi kullanır.
    Veri yoksa :func:`features.make_features` ilgili sütunları NaN bırakır —
    böylece özellik şeması (feature schema) her koşulda aynı kalır.

    Args:
        symbol: Paribu sembolü.
        depth: Her iki taraftan kaç seviye alınacağı.
        snapshot_interval_minutes: Anlık görüntü örnekleme periyodu.
        start: Başlangıç zamanı (UTC).
        end: Bitiş zamanı (UTC).

    Returns:
        :data:`ORDERBOOK_COLUMNS` sütunlu, bar ızgarasına hizalanmış DataFrame.

    Raises:
        NotImplementedError: Her zaman — API bağlantısı henüz yapılmadı.

    Notes:
        Emir defteri anlık görüntüsü **barın kapanış anına** hizalanmalıdır.
        Bar içindeki ortalama defter durumu kullanılırsa, barın kapanışından
        SONRAKİ bilgiyi de içerebileceği için sızıntı riski doğar.
    """
    raise NotImplementedError(
        "TODO: Paribu API sonra bağlanacak. Emir defteri özellikleri NaN kalacak."
    )


def submit_order(
    symbol: str,
    side: Literal["buy", "sell"],
    quantity: float,
    order_type: Literal["market", "limit"] = "market",
    price: float | None = None,
) -> dict[str, object]:
    """Paribu'ya emir gönderir. **(STUB — araştırma aşamasında kullanılmaz)**

    Args:
        symbol: Paribu sembolü.
        side: ``"buy"`` veya ``"sell"``.
        quantity: Baz varlık miktarı.
        order_type: Emir tipi.
        price: Limit emir fiyatı (``order_type="limit"`` ise zorunlu).

    Returns:
        Borsanın emir yanıtı (order id, durum, dolan miktar vb.).

    Raises:
        NotImplementedError: Her zaman — canlı emir gönderimi kapalıdır.
    """
    raise NotImplementedError("TODO: Paribu API sonra bağlanacak. Canlı emir kapalı.")


# --------------------------------------------------------------------------- #
# Yerel veri okuma / yazma
# --------------------------------------------------------------------------- #


def load_ohlcv(
    path: str | Path,
    timestamp_col: str = "timestamp",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Yerel CSV/Parquet dosyasından OHLCV okur ve standart şemaya getirir.

    Args:
        path: ``.csv``, ``.parquet`` veya ``.pq`` uzantılı dosya yolu.
        timestamp_col: Zaman damgası sütununun adı (indeks zaten zamansa yok
            sayılır).
        tz: Hedef saat dilimi. Varsayılan UTC.

    Returns:
        Doğrulanmış, artan sıralı, standart sütunlu OHLCV DataFrame.

    Raises:
        FileNotFoundError: Dosya yoksa.
        ValueError: Uzantı desteklenmiyorsa veya zorunlu sütunlar eksikse.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Desteklenmeyen uzantı: {suffix} (csv/parquet bekleniyor)")

    df.columns = [str(c).strip().lower() for c in df.columns]

    if timestamp_col in df.columns:
        idx = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.drop(columns=[timestamp_col])
        df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df.index.name = "timestamp"
    if tz != "UTC":
        df.index = df.index.tz_convert(tz)

    return validate_ohlcv(df)


def save_ohlcv(df: pd.DataFrame, path: str | Path) -> Path:
    """OHLCV DataFrame'i diske yazar (uzantıya göre CSV veya Parquet).

    Args:
        df: Yazılacak veri.
        path: Hedef dosya yolu.

    Returns:
        Yazılan dosyanın yolu.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index_label="timestamp")
    else:
        df.to_parquet(p, index=True)
    return p


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisini doğrular ve normalize eder.

    Yapılanlar:

    1. Zorunlu sütun kontrolü.
    2. Zaman indeksini artan sıraya sokma ve tekrar eden zaman damgalarını atma
       (son kaydı tutar).
    3. ``high >= max(open, close)`` ve ``low <= min(open, close)`` tutarlılığını
       zorlama (bozuk kayıtlar düzeltilir).
    4. Negatif hacim/fiyat kontrolü.

    Args:
        df: Ham OHLCV verisi.

    Returns:
        Temizlenmiş kopya.

    Raises:
        ValueError: Zorunlu sütunlar eksikse veya indeks zamansal değilse.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV sütunları eksik: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("İndeks DatetimeIndex olmalı.")

    out = df.loc[:, list(OHLCV_COLUMNS)].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0)

    # Tutarsız bar düzeltmesi (bazı borsa uçları bozuk high/low döndürür).
    body_max = out[["open", "close"]].max(axis=1)
    body_min = out[["open", "close"]].min(axis=1)
    out["high"] = out[["high"]].join(body_max.rename("b")).max(axis=1)
    out["low"] = out[["low"]].join(body_min.rename("b")).min(axis=1)

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Fiyat sütunlarında sıfır/negatif değer var.")

    return out


def drop_unclosed_bar(
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Henüz kapanmamış son barı atar.

    CANLI BOTTA ZORUNLU. Borsalar oluşmakta olan barı da döndürür; bu bar
    özelliklere girerse model, henüz gerçekleşmemiş fiyat hareketini kısmen
    görmüş olur (look-ahead bias).

    Args:
        df: OHLCV verisi.
        now: Şu anki zaman (UTC). ``None`` ise sistem saati.
        bar_minutes: Bar periyodu.

    Returns:
        Son barı (kapanmamışsa) atılmış kopya.
    """
    if df.empty:
        return df
    now_ts = pd.Timestamp.utcnow() if now is None else pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    last_close_time = df.index[-1] + pd.Timedelta(minutes=bar_minutes)
    if last_close_time > now_ts:
        return df.iloc[:-1]
    return df


def align_to_bars(
    df: pd.DataFrame,
    bar_minutes: int = 1,
    fill_gaps: bool = False,
) -> pd.DataFrame:
    """Veriyi düzenli bar ızgarasına hizalar.

    Args:
        df: OHLCV verisi.
        bar_minutes: Hedef bar periyodu.
        fill_gaps: ``True`` ise eksik barlar önceki kapanışla doldurulur
            (hacim 0). ``False`` ise eksik barlar NaN kalır.

    Returns:
        Düzenli ızgaraya oturtulmuş DataFrame.

    Notes:
        Boşlukları doldurmak, olmayan likiditeyi varmış gibi gösterir; backtest
        sonuçlarını iyimserleştirebilir. Varsayılan olarak KAPALIDIR.
    """
    freq = f"{bar_minutes}min"
    full_idx = pd.date_range(df.index[0], df.index[-1], freq=freq, tz=df.index.tz)
    out = df.reindex(full_idx)
    out.index.name = "timestamp"
    if fill_gaps:
        out["close"] = out["close"].ffill()
        for col in ("open", "high", "low"):
            out[col] = out[col].fillna(out["close"])
        out["volume"] = out["volume"].fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# Sentetik veri üreteci (test / demo)
# --------------------------------------------------------------------------- #


def generate_synthetic_ohlcv(
    n_bars: int | None = None,
    start: str | pd.Timestamp | None = None,
    start_price: float | None = None,
    annual_vol: float | None = None,
    bar_minutes: int | None = None,
    seed: int = CONFIG.seed,
    drift_annual: float = 0.0,
    momentum_strength: float = 0.05,
    cfg: DataConfig | None = None,
) -> pd.DataFrame:
    """Testler ve uçtan uca demo için sentetik OHLCV üretir.

    Üretim modeli (bilinçli olarak "gerçekçi ama zor" seçilmiştir):

    * Log-fiyat: sürüklenmeli rastgele yürüyüş.
    * Stokastik volatilite: OU sürecine benzer log-vol + günlük mevsimsellik
      (gece/gündüz aktivite farkı) -> rejim değişimi ve vol kümelenmesi.
    * Zayıf momentum bileşeni (``momentum_strength``): modelin bulabileceği
      KÜÇÜK ve gerçekçi bir sinyal. Sıfır yapılırsa veri tamamen öngörülemez
      olur; bu da hattın "hiçbir şey bulamama" davranışını test etmek için
      yararlıdır.
    * Bar içi high/low: Brownian köprü yaklaşımıyla üretilir, böylece
      Parkinson/Garman-Klass tahmincileri anlamlı çalışır.
    * Hacim: volatilite ile pozitif korelasyonlu log-normal.

    Args:
        n_bars: Üretilecek bar sayısı.
        start: Başlangıç zamanı (UTC).
        start_price: Başlangıç fiyatı.
        annual_vol: Hedeflenen yıllık volatilite.
        bar_minutes: Bar periyodu.
        seed: Rastgelelik tohumu (tekrarlanabilirlik).
        drift_annual: Yıllık sürüklenme (drift).
        momentum_strength: Getirilerdeki AR(1) benzeri kalıcılık katsayısı.
        cfg: Varsayılanların okunacağı :class:`DataConfig`.

    Returns:
        Standart şemada, doğrulanmış OHLCV DataFrame.
    """
    c = cfg or CONFIG.data
    n = int(n_bars if n_bars is not None else c.synthetic_bars)
    bm = int(bar_minutes if bar_minutes is not None else c.bar_minutes)
    p0 = float(start_price if start_price is not None else c.synthetic_start_price)
    av = float(annual_vol if annual_vol is not None else c.synthetic_annual_vol)
    t0 = pd.Timestamp(start if start is not None else c.synthetic_start, tz="UTC")

    rng = np.random.default_rng(seed)
    bars_per_year = 365.0 * 24.0 * 60.0 / bm
    dt = 1.0 / bars_per_year
    base_sigma = av * np.sqrt(dt)  # bar başına baz volatilite

    index = pd.date_range(t0, periods=n, freq=f"{bm}min", tz="UTC")

    # --- Stokastik volatilite (log-vol üzerinde OU) --------------------------
    kappa, vol_of_vol = 0.002, 0.06
    log_vol = np.zeros(n)
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        log_vol[i] = (1.0 - kappa) * log_vol[i - 1] + vol_of_vol * shocks[i]

    # Günlük mevsimsellik: UTC 12:00-20:00 arası daha hareketli.
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    seasonal = 1.0 + 0.35 * np.sin(2.0 * np.pi * (hour - 6.0) / 24.0)
    sigma = base_sigma * np.exp(log_vol) * seasonal

    # --- Getiriler: zayıf AR(1) momentum + gürültü --------------------------
    eps = rng.standard_normal(n) * sigma
    rets = np.zeros(n)
    mu = drift_annual * dt
    for i in range(1, n):
        rets[i] = mu + momentum_strength * rets[i - 1] + eps[i]

    log_close = np.log(p0) + np.cumsum(rets)
    close = np.exp(log_close)
    open_ = np.empty(n)
    open_[0] = p0
    open_[1:] = close[:-1]

    # --- Bar içi ekstremler (Brownian köprü yaklaşımı) ----------------------
    # Bar içinde ~sigma ölçeğinde ek salınım; high/low daima gövdeyi kapsar.
    u = rng.random(n)
    v = rng.random(n)
    # Beklenen menzil için ölçek: |N(0,1)| benzeri pozitif çekilişler.
    up_ext = sigma * np.sqrt(-2.0 * np.log(np.clip(u, 1e-12, 1.0))) * 0.5
    dn_ext = sigma * np.sqrt(-2.0 * np.log(np.clip(v, 1e-12, 1.0))) * 0.5
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(up_ext)
    low = body_lo * np.exp(-dn_ext)

    # --- Hacim: volatilite ile korelasyonlu log-normal -----------------------
    vol_noise = rng.standard_normal(n) * 0.4
    rel_sigma = sigma / base_sigma
    volume = np.exp(1.0 + 0.8 * np.log(rel_sigma) + vol_noise) * 10.0

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return validate_ohlcv(df)


def generate_synthetic_reference(
    base: pd.DataFrame,
    correlation: float = 0.75,
    lead_bars: int = 3,
    seed: int = CONFIG.seed + 1,
) -> pd.DataFrame:
    """Ana sembolle korelasyonlu, ONU ÖNCELEYEN sentetik referans (BTC) serisi.

    Çapraz varlık özelliklerini (BTC öncülüğü) test etmek için kullanılır.
    Referans serinin getirileri, ana serinin gelecekteki getirileriyle
    ilişkilendirilir; yani referans ``lead_bars`` kadar ÖNDE gider.

    Args:
        base: Ana sembolün OHLCV verisi.
        correlation: Hedeflenen korelasyon (0-1).
        lead_bars: Referansın kaç bar önde olduğu.
        seed: Rastgelelik tohumu.

    Returns:
        Ana veriyle aynı indekse sahip OHLCV DataFrame.

    Notes:
        Bu fonksiyon SADECE sentetik test verisi üretir. ``lead_bars`` ileri
        kaydırma burada kasıtlıdır (referans önde gitsin diye); özellik
        üretiminde bu tür ileri kaydırma ASLA yapılmaz.
    """
    rng = np.random.default_rng(seed)
    base_ret = np.log(base["close"]).diff().fillna(0.0).to_numpy()
    # Referans, ana serinin GELECEK getirisini kısmen "önceden" içerir.
    led = np.roll(base_ret, -lead_bars)
    led[-lead_bars:] = 0.0
    noise = rng.standard_normal(len(base_ret)) * base_ret.std()
    ref_ret = correlation * led + np.sqrt(max(1e-12, 1.0 - correlation**2)) * noise

    close = 100_000.0 * np.exp(np.cumsum(ref_ret))
    open_ = np.concatenate([[100_000.0], close[:-1]])
    spread = np.abs(ref_ret) * close
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.abs(rng.standard_normal(len(close))) * 100.0 + 1.0,
        },
        index=base.index,
    )
    df.index.name = "timestamp"
    return validate_ohlcv(df)


def generate_synthetic_orderbook(
    base: pd.DataFrame,
    seed: int = CONFIG.seed + 2,
) -> pd.DataFrame:
    """Mikroyapı özelliklerini test etmek için sentetik emir defteri üretir.

    Gerçek defter verisi geldiğinde bu fonksiyon devre dışı kalır; şema
    :data:`ORDERBOOK_COLUMNS` ile birebir aynıdır.

    Args:
        base: Ana OHLCV verisi (fiyat ve volatilite referansı).
        seed: Rastgelelik tohumu.

    Returns:
        Bar ızgarasına hizalanmış emir defteri anlık görüntüleri.
    """
    rng = np.random.default_rng(seed)
    n = len(base)
    close = base["close"].to_numpy()
    ret = np.abs(np.log(base["close"]).diff().fillna(0.0).to_numpy())
    # Spread volatilite ile genişler (gerçek piyasa davranışı).
    half_spread = close * (2e-4 + 5.0 * ret) * 0.5
    bid = close - half_spread
    ask = close + half_spread

    imbalance = np.clip(rng.standard_normal(n) * 0.3, -0.9, 0.9)
    total_size = np.exp(rng.standard_normal(n) * 0.5 + 1.0)
    bid_size = total_size * (1.0 + imbalance) / 2.0
    ask_size = total_size * (1.0 - imbalance) / 2.0

    df = pd.DataFrame(
        {
            "bid_price": bid,
            "ask_price": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "bid_depth": bid_size * (5.0 + rng.random(n) * 5.0),
            "ask_depth": ask_size * (5.0 + rng.random(n) * 5.0),
        },
        index=base.index,
    )
    df.index.name = "timestamp"
    return df


def build_synthetic_dataset(
    cfg: ResearchConfig | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Uçtan uca demo için eksiksiz sentetik veri paketi üretir.

    Args:
        cfg: Kök konfigürasyon.

    Returns:
        ``{"ohlcv", "reference", "orderbook", "stable_premium"}`` anahtarlı
        sözlük. ``stable_premium`` USDT/TL primi için örnek seridir.
    """
    c = cfg or CONFIG
    ohlcv = generate_synthetic_ohlcv(cfg=c.data, seed=c.seed)
    reference = generate_synthetic_reference(ohlcv, seed=c.seed + 1)
    orderbook = generate_synthetic_orderbook(ohlcv, seed=c.seed + 2)

    rng = np.random.default_rng(c.seed + 3)
    # USDT/TL primi: yavaş hareket eden, ortalamaya dönen küçük bir seri.
    prem = np.cumsum(rng.standard_normal(len(ohlcv)) * 1e-4)
    prem = pd.Series(prem - pd.Series(prem).rolling(1440, min_periods=1).mean().to_numpy(),
                     index=ohlcv.index, name="stable_premium")
    return {
        "ohlcv": ohlcv,
        "reference": reference,
        "orderbook": orderbook,
        "stable_premium": prem,
    }
