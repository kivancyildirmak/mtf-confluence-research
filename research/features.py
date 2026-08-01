"""ORTAK özellik mühendisliği modülü — hattın kalbi.

**FEATURE PARITY (en kritik kural).** Backtest de, ileride yazılacak canlı bot
da özellikleri SADECE buradaki :func:`make_features` ile üretmelidir. İki taraf
farklı kod yolları kullanırsa, backtest'te görülen performans canlıda asla
tekrar etmez. Bu yüzden bu modül:

* Girdi olarak yalnızca standart OHLCV (+ opsiyonel emir defteri / referans
  varlık) alır,
* Deterministiktir (rastgelelik yok),
* Global durum (state) tutmaz,
* Eksik veri kaynaklarında sütunları SİLMEZ, NaN bırakır — böylece özellik
  şeması her koşulda birebir aynıdır.

**SIZINTI (LOOK-AHEAD) POLİTİKASI.**
Tüm hesaplar "bar ``t`` kapandıktan sonra bilinen bilgi" ile sınırlıdır:

* Sadece geriye bakan (trailing) ``rolling`` / ``ewm`` pencereleri kullanılır.
* ``shift(-k)`` (ileri kaydırma), ``center=True`` pencere, tüm örnekle
  hesaplanan ortalama/std (global standardizasyon) ve ``bfill`` **YASAKTIR**.
* Bar ``t``'nin kendi ``close``'u özelliklere girer; bu meşrudur çünkü bar
  kapanmıştır. Ancak işlem bar ``t``'de değil, ``BacktestConfig.latency_bars``
  kadar sonraki barın AÇILIŞINDA gerçekleşir (bkz. :mod:`backtest`).
* Bar ``t``'nin ``high``/``low``'u da bar kapandığında bilinir; fakat bunları
  BARIN İÇİNDE karar vermek için kullanmak sızıntıdır. Backtest bunu asla
  yapmaz.

Her özellik grubunun başında, o gruba özgü sızıntı riski yorumda belirtilmiştir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, FeatureConfig

#: Emir defteri verisi yoksa NaN olarak yine de üretilecek mikroyapı sütunları.
MICROSTRUCTURE_COLUMNS: tuple[str, ...] = (
    "mk_spread_rel",
    "mk_spread_z",
    "mk_imbalance",
    "mk_imbalance_ma",
    "mk_depth_total",
    "mk_depth_ratio",
    "mk_micro_price_dev",
)

#: Çapraz varlık verisi yoksa NaN olarak üretilecek sütunlar (placeholder).
CROSS_ASSET_STATIC_COLUMNS: tuple[str, ...] = (
    "xa_stable_premium",
    "xa_stable_premium_z",
)


# --------------------------------------------------------------------------- #
# Küçük yardımcılar
# --------------------------------------------------------------------------- #


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Sıfıra bölmeyi NaN'a çeviren güvenli bölme."""
    return a / b.replace(0.0, np.nan)


def _rolling_z(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Kayan (trailing) z-skoru.

    SIZINTI NOTU: Standardizasyon global ortalama/std ile YAPILMAZ; global
    istatistik geleceği içerir. Burada yalnızca geçmiş pencere kullanılır.

    Args:
        series: Girdi serisi.
        window: Pencere uzunluğu (bar).
        min_periods: Minimum gözlem sayısı.

    Returns:
        Z-skoru serisi.
    """
    mp = min_periods or max(2, window // 2)
    roll = series.rolling(window, min_periods=mp)
    return _safe_div(series - roll.mean(), roll.std(ddof=0))


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder RSI'ı (0-100) sürekli değişken olarak hesaplar.

    İkili sinyale (>70 / <30) ÇEVRİLMEZ; ham değer modele verilir, eşikleme
    kararını ağaçlar kendisi öğrenir.

    Args:
        close: Kapanış serisi.
        window: RSI periyodu.

    Returns:
        RSI serisi.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_histogram(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series]:
    """MACD çizgisini ve histogramını döndürür.

    Args:
        close: Kapanış serisi.
        fast: Hızlı EMA periyodu.
        slow: Yavaş EMA periyodu.
        signal: Sinyal EMA periyodu.

    Returns:
        ``(macd_line, histogram)`` ikilisi. Fiyat seviyesinden bağımsız olması
        için ikisi de kapanışa oranlanır.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return _safe_div(macd, close), _safe_div(macd - sig, close)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range (Wilder yumuşatması), kapanışa oranlanmış.

    Args:
        df: OHLCV verisi.
        window: ATR periyodu.

    Returns:
        Göreli ATR serisi.
    """
    prev_close = df["close"].shift(1)  # shift(1): geçmişe bakış, güvenli.
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return _safe_div(atr, df["close"])


def _parkinson_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson volatilite tahmincisi (high-low aralığına dayalı).

    Kapanış-kapanış tahmincisine göre daha verimlidir ama sıçramaları (gap)
    yakalayamaz.

    Args:
        df: OHLCV verisi.
        window: Pencere uzunluğu.

    Returns:
        Bar başına volatilite tahmini.
    """
    hl = np.log(_safe_div(df["high"], df["low"]))
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * (hl**2).rolling(window, min_periods=max(2, window // 2)).mean())


def _garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass volatilite tahmincisi (OHLC'nin tamamını kullanır).

    Args:
        df: OHLCV verisi.
        window: Pencere uzunluğu.

    Returns:
        Bar başına volatilite tahmini.
    """
    hl = np.log(_safe_div(df["high"], df["low"])) ** 2
    co = np.log(_safe_div(df["close"], df["open"])) ** 2
    est = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    mean = est.rolling(window, min_periods=max(2, window // 2)).mean()
    return np.sqrt(mean.clip(lower=0.0))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (kümülatif işaretli hacim).

    Args:
        close: Kapanış serisi.
        volume: Hacim serisi.

    Returns:
        OBV serisi (ham, seviye olarak durağan değildir).
    """
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


# --------------------------------------------------------------------------- #
# Özellik grupları
# --------------------------------------------------------------------------- #


def _return_features(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Getiri türevleri: çok ufuklu log-getiriler, kayan ortalama ve std.

    SIZINTI NOTU: ``diff(h)`` geriye bakar (``x[t] - x[t-h]``), güvenlidir.
    """
    out: dict[str, pd.Series] = {}
    log_close = np.log(df["close"])
    r1 = log_close.diff()
    out["ret_1"] = r1

    for h in cfg.return_horizons:
        if h != 1:
            out[f"ret_{h}"] = log_close.diff(h)
        # Ufka göre normalize edilmiş getiri (sqrt-zaman ölçeklemesi).
        out[f"ret_{h}_norm"] = log_close.diff(h) / np.sqrt(h)

    for w in cfg.ma_windows:
        mp = max(2, w // 2)
        out[f"ret_mean_{w}"] = r1.rolling(w, min_periods=mp).mean()
        out[f"ret_std_{w}"] = r1.rolling(w, min_periods=mp).std(ddof=0)
        # Getiri işaretinin kalıcılığı (trend gücü göstergesi).
        out[f"ret_sign_mean_{w}"] = np.sign(r1).rolling(w, min_periods=mp).mean()
    return out


def _volatility_features(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Volatilite ailesi: gerçekleşmiş vol, ATR, Parkinson, Garman-Klass, vol-of-vol.

    SIZINTI NOTU: Tüm tahminciler yalnızca ``t``'ye kadar kapanmış barları
    kullanır. Bar ``t``'nin high/low'u dahildir (bar kapandı => bilinir).
    """
    out: dict[str, pd.Series] = {}
    r1 = np.log(df["close"]).diff()

    for w in cfg.vol_windows:
        mp = max(2, w // 2)
        rv = r1.rolling(w, min_periods=mp).std(ddof=0)
        out[f"vol_rv_{w}"] = rv
        out[f"vol_pk_{w}"] = _parkinson_vol(df, w)
        out[f"vol_gk_{w}"] = _garman_klass_vol(df, w)
        # Tahminciler arası oran: sıçrama (gap) rejimini ele verir.
        out[f"vol_pk_rv_ratio_{w}"] = _safe_div(out[f"vol_pk_{w}"], rv)

    for w in cfg.atr_windows:
        out[f"vol_atr_{w}"] = _atr(df, w)

    # Vol-of-vol: volatilitenin kendi oynaklığı (rejim geçişlerine duyarlı).
    base_w = cfg.vol_windows[0]
    base_vol = r1.rolling(base_w, min_periods=max(2, base_w // 2)).std(ddof=0)
    vov_w = cfg.vol_of_vol_window
    out["vol_of_vol"] = base_vol.rolling(vov_w, min_periods=max(2, vov_w // 4)).std(ddof=0)
    out["vol_of_vol_rel"] = _safe_div(out["vol_of_vol"], base_vol)
    # Kısa/uzun vol oranı: vol genişliyor mu daralıyor mu?
    long_w = cfg.vol_windows[-1]
    out["vol_term_ratio"] = _safe_div(
        base_vol, r1.rolling(long_w, min_periods=max(2, long_w // 2)).std(ddof=0)
    )
    return out


def _momentum_features(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Momentum ailesi — hepsi SÜREKLİ değişken olarak (ikili sinyal üretilmez).

    SIZINTI NOTU: EMA'lar ``adjust=False`` ile özyinelemeli hesaplanır; her
    değer yalnızca geçmişe bağlıdır. ``min_periods`` sayesinde ısınma
    (warm-up) süresi dolmadan değer üretilmez.
    """
    out: dict[str, pd.Series] = {}
    close = df["close"]

    for w in cfg.rsi_windows:
        # RSI'ı 0-100 yerine -1..+1 civarına taşımak ağaçlar için şart değil,
        # ama diğer özelliklerle ölçek uyumunu kolaylaştırır.
        out[f"mom_rsi_{w}"] = _wilder_rsi(close, w)
        out[f"mom_rsi_{w}_c"] = (out[f"mom_rsi_{w}"] - 50.0) / 50.0

    fast, slow, sig = cfg.macd_params
    macd_line, macd_hist = _macd_histogram(close, fast, slow, sig)
    out["mom_macd"] = macd_line
    out["mom_macd_hist"] = macd_hist
    out["mom_macd_hist_slope"] = macd_hist.diff()

    for w in cfg.roc_windows:
        out[f"mom_roc_{w}"] = close.pct_change(w)

    for w in cfg.zscore_windows:
        mp = max(2, w // 2)
        ma = close.rolling(w, min_periods=mp).mean()
        sd = close.rolling(w, min_periods=mp).std(ddof=0)
        out[f"mom_price_z_{w}"] = _safe_div(close - ma, sd)
        out[f"mom_price_ma_dev_{w}"] = _safe_div(close - ma, ma)

    # Kısa/uzun ortalama farkı (trend eğimi), fiyata oranlanmış.
    if len(cfg.ma_windows) >= 2:
        short_ma = close.rolling(cfg.ma_windows[0], min_periods=2).mean()
        long_ma = close.rolling(cfg.ma_windows[-1], min_periods=2).mean()
        out["mom_ma_spread"] = _safe_div(short_ma - long_ma, close)
    return out


def _volume_features(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Hacim ailesi: hacim z-skoru, VWAP sapması, OBV, hacim-ağırlıklı momentum.

    SIZINTI NOTU: VWAP kayan pencerede hesaplanır. Günlük (seans başından
    itibaren) VWAP kullanılacaksa, gün içi ilerledikçe pencere büyür ama yine
    geçmişe bakar — o da güvenlidir. Burada kayan pencere tercih edilmiştir.
    """
    out: dict[str, pd.Series] = {}
    volume = df["volume"]
    close = df["close"]
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    r1 = np.log(close).diff()

    for w in cfg.volume_windows:
        mp = max(2, w // 2)
        out[f"volm_z_{w}"] = _rolling_z(volume, w, mp)
        out[f"volm_ratio_{w}"] = _safe_div(volume, volume.rolling(w, min_periods=mp).mean())
        # Hacim-ağırlıklı momentum: büyük hacimli barlara daha çok ağırlık.
        vw = (r1 * volume).rolling(w, min_periods=mp).sum()
        out[f"volm_weighted_mom_{w}"] = _safe_div(vw, volume.rolling(w, min_periods=mp).sum())

    for w in cfg.vwap_windows:
        mp = max(2, w // 2)
        pv = (typical * volume).rolling(w, min_periods=mp).sum()
        vv = volume.rolling(w, min_periods=mp).sum()
        vwap = _safe_div(pv, vv)
        out[f"volm_vwap_dev_{w}"] = _safe_div(close - vwap, vwap)

    obv = _obv(close, volume)
    out["volm_obv"] = obv
    # Ham OBV durağan değildir; modele normalize edilmiş türevlerini de veriyoruz.
    base_w = cfg.volume_windows[-1]
    out["volm_obv_z"] = _rolling_z(obv, base_w)
    out["volm_obv_slope"] = _safe_div(
        obv.diff(base_w), volume.rolling(base_w, min_periods=2).sum()
    )
    return out


def _microstructure_features(
    df: pd.DataFrame,
    orderbook: pd.DataFrame | None,
    cfg: FeatureConfig,
) -> dict[str, pd.Series]:
    """Mikroyapı ailesi: spread, emir defteri dengesizliği, derinlik.

    Emir defteri verisi yoksa TÜM sütunlar NaN olarak üretilir — özellik şeması
    bozulmasın diye. LightGBM NaN'ı doğal olarak işler.

    SIZINTI NOTU: Anlık görüntü barın KAPANIŞ anına ait olmalıdır. Bar içi
    ortalama defter kullanılırsa, karar anından sonraki bilgiyi içerebilir.
    """
    idx = df.index
    if orderbook is None or orderbook.empty:
        nan = pd.Series(np.nan, index=idx, dtype="float64")
        return {name: nan.copy() for name in MICROSTRUCTURE_COLUMNS}

    ob = orderbook.reindex(idx)  # reindex: ileri doldurma YOK, hizalama var.
    bid, ask = ob["bid_price"], ob["ask_price"]
    bid_sz, ask_sz = ob["bid_size"], ob["ask_size"]
    mid = (bid + ask) / 2.0

    out: dict[str, pd.Series] = {}
    out["mk_spread_rel"] = _safe_div(ask - bid, mid)
    out["mk_spread_z"] = _rolling_z(out["mk_spread_rel"], cfg.ma_windows[-1])
    out["mk_imbalance"] = _safe_div(bid_sz - ask_sz, bid_sz + ask_sz)
    out["mk_imbalance_ma"] = out["mk_imbalance"].rolling(
        cfg.ma_windows[1], min_periods=2
    ).mean()

    bid_depth = ob["bid_depth"] if "bid_depth" in ob else bid_sz
    ask_depth = ob["ask_depth"] if "ask_depth" in ob else ask_sz
    out["mk_depth_total"] = np.log1p(bid_depth + ask_depth)
    out["mk_depth_ratio"] = _safe_div(bid_depth - ask_depth, bid_depth + ask_depth)

    # Mikro-fiyat (hacim ağırlıklı mid) ile son işlem fiyatı arasındaki sapma:
    # kısa vadeli yön baskısının klasik göstergesi.
    micro = _safe_div(bid * ask_sz + ask * bid_sz, bid_sz + ask_sz)
    out["mk_micro_price_dev"] = _safe_div(micro - df["close"], df["close"])
    return out


def _cross_asset_features(
    df: pd.DataFrame,
    reference: pd.DataFrame | None,
    stable_premium: pd.Series | None,
    cfg: FeatureConfig,
) -> dict[str, pd.Series]:
    """Çapraz varlık: BTC öncülüğü/gecikmesi, korelasyon, USDT/TL primi.

    SIZINTI NOTU — BU GRUP EN RİSKLİ OLANI:

    * Referans varlığın bar ``t`` getirisi ancak o bar KAPANDIĞINDA bilinir.
      Aynı ``t`` barında karar verilecekse bu meşrudur (biz de bar sonunda karar
      veriyoruz), fakat canlıda referans borsanın bar kapanışı geç gelirse
      pratikte elde olmayan veriyle çalışılmış olur. Bu yüzden öncü (lead)
      özellikleri ``shift(lag)`` ile GECİKTİRİLMİŞ olarak da üretiyoruz.
    * ``shift(-k)`` (referansın GELECEĞİ) asla kullanılmaz.
    """
    idx = df.index
    out: dict[str, pd.Series] = {}
    r1 = np.log(df["close"]).diff()

    if reference is None or reference.empty:
        nan = pd.Series(np.nan, index=idx, dtype="float64")
        out["xa_ref_ret_1"] = nan.copy()
        for lag in cfg.lead_lag_bars:
            out[f"xa_ref_ret_lag_{lag}"] = nan.copy()
            out[f"xa_ref_lead_diff_{lag}"] = nan.copy()
        for w in cfg.corr_windows:
            out[f"xa_ref_corr_{w}"] = nan.copy()
            out[f"xa_ref_beta_{w}"] = nan.copy()
    else:
        ref_close = reference["close"].reindex(idx)
        ref_r1 = np.log(ref_close).diff()
        out["xa_ref_ret_1"] = ref_r1
        for lag in cfg.lead_lag_bars:
            # Referansın GEÇMİŞ getirisi: BTC öncülüğü hipotezinin testi.
            out[f"xa_ref_ret_lag_{lag}"] = ref_r1.shift(lag)
            # Referans ile ana varlığın kümülatif getiri farkı (yayılma/spread).
            out[f"xa_ref_lead_diff_{lag}"] = (
                ref_r1.rolling(lag, min_periods=1).sum()
                - r1.rolling(lag, min_periods=1).sum()
            )
        for w in cfg.corr_windows:
            mp = max(3, w // 2)
            out[f"xa_ref_corr_{w}"] = r1.rolling(w, min_periods=mp).corr(ref_r1)
            cov = r1.rolling(w, min_periods=mp).cov(ref_r1)
            var = ref_r1.rolling(w, min_periods=mp).var(ddof=0)
            out[f"xa_ref_beta_{w}"] = _safe_div(cov, var)

    if stable_premium is None:
        nan = pd.Series(np.nan, index=idx, dtype="float64")
        out["xa_stable_premium"] = nan.copy()
        out["xa_stable_premium_z"] = nan.copy()
    else:
        prem = pd.Series(stable_premium).reindex(idx).astype("float64")
        out["xa_stable_premium"] = prem
        out["xa_stable_premium_z"] = _rolling_z(prem, cfg.corr_windows[-1])
    return out


def _regime_calendar_features(df: pd.DataFrame, cfg: FeatureConfig) -> dict[str, pd.Series]:
    """Rejim ve takvim: saat/gün, volatilite rejimi bayrağı, kayan skew/kurtosis.

    SIZINTI NOTU: Takvim özellikleri (saat, gün) gelecekte de bilinir; sızıntı
    yaratmazlar. Rejim bayrağı ise UZUN pencereli kayan medyana göre
    hesaplanır — sabit bir eşik "tüm örneğe bakarak" seçilirse sızıntı olur,
    o yüzden eşik de kayan pencereden gelir.
    """
    out: dict[str, pd.Series] = {}
    idx = df.index
    r1 = np.log(df["close"]).diff()

    hour = pd.Series(idx.hour.to_numpy(), index=idx, dtype="float64")
    dow = pd.Series(idx.dayofweek.to_numpy(), index=idx, dtype="float64")
    minute_of_day = hour * 60.0 + pd.Series(idx.minute.to_numpy(), index=idx, dtype="float64")

    out["cal_hour"] = hour
    out["cal_dayofweek"] = dow
    out["cal_is_weekend"] = (dow >= 5).astype("float64")
    # Döngüsel kodlama: 23:59 ile 00:00 arasındaki yapay uçurumu kaldırır.
    out["cal_tod_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    out["cal_tod_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    out["cal_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["cal_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    short_w = cfg.vol_windows[0]
    vol_short = r1.rolling(short_w, min_periods=max(2, short_w // 2)).std(ddof=0)
    ref = vol_short.rolling(cfg.regime_window, min_periods=cfg.regime_window // 4).median()
    out["reg_vol_ratio"] = _safe_div(vol_short, ref)
    out["reg_high_vol"] = (out["reg_vol_ratio"] > cfg.regime_high_mult).astype("float64")
    out["reg_low_vol"] = (out["reg_vol_ratio"] < 1.0 / cfg.regime_high_mult).astype("float64")
    # Trend rejimi: uzun pencerede getiri işaretinin tutarlılığı.
    out["reg_trend_strength"] = _safe_div(
        r1.rolling(cfg.regime_window, min_periods=cfg.regime_window // 4).mean(),
        r1.rolling(cfg.regime_window, min_periods=cfg.regime_window // 4).std(ddof=0),
    )

    for w in cfg.moment_windows:
        mp = max(4, w // 2)
        out[f"reg_skew_{w}"] = r1.rolling(w, min_periods=mp).skew()
        out[f"reg_kurt_{w}"] = r1.rolling(w, min_periods=mp).kurt()
    return out


# --------------------------------------------------------------------------- #
# Ana giriş noktası
# --------------------------------------------------------------------------- #


def make_features(
    df: pd.DataFrame,
    orderbook: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    stable_premium: pd.Series | None = None,
    cfg: FeatureConfig | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Tüm özellik gruplarını üreten TEK giriş noktası.

    Canlı bot da, backtest de yalnızca bu fonksiyonu çağırmalıdır.

    Args:
        df: Standart OHLCV DataFrame (bkz. :mod:`data`). Yalnızca KAPANMIŞ
            barlar içermelidir.
        orderbook: Opsiyonel emir defteri anlık görüntüleri. ``None`` ise
            mikroyapı sütunları NaN üretilir.
        reference: Opsiyonel referans varlık OHLCV'si (örn. BTC/USDT).
        stable_premium: Opsiyonel USDT/TL prim serisi.
        cfg: Özellik konfigürasyonu. ``None`` ise :data:`config.CONFIG`.
        dropna: ``True`` ise ısınma (warm-up) döneminden kaynaklanan NaN
            satırları atılır. Araştırmada ``False`` bırakıp hizalamayı
            :mod:`labeling` tarafında yapmak daha güvenlidir.

    Returns:
        Girdi ile aynı indekse sahip, yalnızca ÖZELLİK sütunları içeren
        DataFrame. Sütun sırası deterministiktir (parite için kritik).

    Raises:
        ValueError: Girdi şeması bozuksa veya indeks artan sıralı değilse.
    """
    c = cfg or CONFIG.features

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"make_features için eksik sütunlar: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("make_features DatetimeIndex bekler.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("İndeks artan sıralı olmalı (aksi halde rolling anlamsızlaşır).")
    if df.index.has_duplicates:
        raise ValueError("İndekste tekrar eden zaman damgaları var.")

    groups: dict[str, pd.Series] = {}
    groups.update(_return_features(df, c))
    groups.update(_volatility_features(df, c))
    groups.update(_momentum_features(df, c))
    groups.update(_volume_features(df, c))
    groups.update(_microstructure_features(df, orderbook, c))
    groups.update(_cross_asset_features(df, reference, stable_premium, c))
    groups.update(_regime_calendar_features(df, c))

    out = pd.DataFrame(groups, index=df.index)
    out = out.astype("float64")

    # Sonsuz değerleri temizle ve uç değerleri kırp (ağaçlar için zararsız,
    # ama SHAP/grafiklerde sayısal taşmayı önler).
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.clip(lower=-c.clip_abs, upper=c.clip_abs)

    # Sütun sırasını sabitle: modelin beklediği şema ile canlı şema aynı olmalı.
    out = out.reindex(columns=sorted(out.columns))
    out.index.name = df.index.name or "timestamp"

    if dropna:
        out = out.dropna(how="any")
    return out


def feature_names(cfg: FeatureConfig | None = None) -> list[str]:
    """Üretilecek özellik sütunlarının adlarını (sıralı) döndürür.

    Küçük bir kukla veri üzerinde :func:`make_features` çalıştırarak şemayı
    türetir; böylece isim listesi ile üretim kodu asla ayrışmaz.

    Args:
        cfg: Özellik konfigürasyonu.

    Returns:
        Alfabetik sıralı özellik adları listesi.
    """
    c = cfg or CONFIG.features
    n = max(
        max(c.ma_windows), max(c.vol_windows), c.regime_window, c.vol_of_vol_window
    ) + 10
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    dummy = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.001,
            "low": 0.999,
            "close": 1.0,
            "volume": 1.0,
        },
        index=idx,
    )
    return list(make_features(dummy, cfg=c).columns)


def warmup_bars(cfg: FeatureConfig | None = None) -> int:
    """Özelliklerin güvenilir hale gelmesi için gereken minimum bar sayısı.

    Canlı botta, elde bu kadar geçmiş yoksa sinyal üretilmemelidir.

    Args:
        cfg: Özellik konfigürasyonu.

    Returns:
        Isınma için gereken bar sayısı.
    """
    c = cfg or CONFIG.features
    return int(
        max(
            max(c.ma_windows),
            max(c.vol_windows),
            max(c.atr_windows),
            max(c.volume_windows),
            max(c.vwap_windows),
            max(c.corr_windows),
            max(c.moment_windows),
            c.vol_of_vol_window,
            c.regime_window,
        )
    )
