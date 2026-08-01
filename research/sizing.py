"""Pozisyon boyutlandırma: olasılık -> bahis, kapaklı Kelly, volatilite hedefleme.

Ayrı bir modül olmasının sebebi: **yön kararı ile büyüklük kararı farklı
problemlerdir.** Model "gireyim mi?" sorusunu olasılıkla cevaplar; buradaki
fonksiyonlar "ne kadar?" sorusunu cevaplar. Canlı bot da backtest de aynı
boyutlandırmayı kullanmak zorundadır (feature parity'nin kardeşi: sizing parity).

Üç bileşen birleştirilir ve **en muhafazakârı** seçilir:

1. **Olasılık boyutu** — güven arttıkça pozisyon büyür.
2. **Kapaklı (fractional) Kelly** — teorik optimum, ama tam Kelly pratikte
   yıkıcıdır (tahmin hatasına aşırı duyarlı), bu yüzden kesirli ve kapaklı.
3. **Volatilite hedefleme** — portföy riskini sabit tutar; yüksek vol
   rejimlerinde pozisyonu otomatik küçültür.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import CONFIG, SizingConfig

ArrayLike = np.ndarray | pd.Series | float


def _as_array(x: ArrayLike) -> np.ndarray:
    """Girdiyi float NumPy dizisine çevirir."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype="float64")
    return np.atleast_1d(np.asarray(x, dtype="float64"))


def prob_to_bet_size(
    prob: ArrayLike,
    method: str = "zscore",
    scale: float = 2.0,
) -> np.ndarray:
    """Model olasılığını 0-1 arası bahis büyüklüğüne çevirir.

    Args:
        prob: Pozitif sınıf olasılığı (meta-model çıktısı).
        method: ``"linear"`` -> ``scale * (p - 0.5)``;
            ``"zscore"`` -> López de Prado dönüşümü
            ``2 * Phi((p-0.5)/sqrt(p(1-p))) - 1``.
        scale: Doğrusal yöntemin eğimi.

    Returns:
        ``[0, 1]`` aralığında bahis büyüklükleri (yön BİLGİSİ İÇERMEZ).

    Notes:
        ``p = 0.5`` -> boyut 0 (bilgi yok). ``zscore`` yöntemi 0.5 civarında
        daha keskin, uçlarda daha yumuşaktır; kalibrasyonu bozuk modellerde
        aşırı pozisyonu engeller.
    """
    p = np.clip(_as_array(prob), 1e-6, 1.0 - 1e-6)
    if method == "linear":
        size = scale * (p - 0.5)
    elif method == "zscore":
        z = (p - 0.5) / np.sqrt(p * (1.0 - p))
        size = 2.0 * norm.cdf(z) - 1.0
    else:
        raise ValueError(f"Bilinmeyen method: {method}")
    return np.clip(np.abs(size), 0.0, 1.0)


def kelly_size(
    prob: ArrayLike,
    payoff_ratio: float = 1.0,
    fraction: float = 0.25,
    cap: float = 0.5,
) -> np.ndarray:
    """Kesirli (fractional) ve kapaklı Kelly bahis oranı.

    Kelly formülü (ikili sonuç):  ``f* = (p*b - (1-p)) / b``

    Args:
        prob: Kazanma olasılığı.
        payoff_ratio: ``b`` = ortalama kazanç / ortalama kayıp. Triple-barrier'da
            bariyer oranından (``pt / sl``) türetilebilir.
        fraction: Kelly kesri (0.25 = çeyrek Kelly).
        cap: Uygulanacak üst sınır.

    Returns:
        ``[0, cap]`` aralığında pozisyon oranı.

    Notes:
        Tam Kelly (``fraction=1``) **önerilmez**: ``p`` tahminindeki küçük bir
        hata bile iflas riskini hızla büyütür. Ayrıca Kelly, getirilerin i.i.d.
        olduğunu varsayar — kripto piyasasında bu varsayım zayıftır.
    """
    p = np.clip(_as_array(prob), 0.0, 1.0)
    b = max(float(payoff_ratio), 1e-6)
    f = (p * b - (1.0 - p)) / b
    f = np.clip(f, 0.0, None) * float(fraction)
    return np.clip(f, 0.0, float(cap))


def vol_target_size(
    vol_forecast: ArrayLike,
    target_annual_vol: float = 0.20,
    bars_per_year: float = 525_600.0,
    holding_bars: int = 60,
    max_leverage: float = 3.0,
) -> np.ndarray:
    """Volatilite hedeflemesine göre kaldıraç katsayısı.

    ``pozisyon = hedef_vol / tahmini_vol`` (aynı ufka ölçeklenmiş).

    Args:
        vol_forecast: BAR başına volatilite tahmini (oran).
        target_annual_vol: Yıllık hedef volatilite.
        bars_per_year: Yıl başına bar sayısı.
        holding_bars: Tipik tutma süresi (ölçekleme için).
        max_leverage: Üst kaldıraç sınırı.

    Returns:
        ``[0, max_leverage]`` aralığında ölçek katsayısı.

    Notes:
        SIZINTI: ``vol_forecast`` mutlaka KAYAN pencereden gelmelidir. Tüm
        örneğin volatilitesiyle ölçekleme yapmak, geleceği bilerek risk ayarı
        yapmak demektir ve backtest'i ciddi biçimde iyimserleştirir.
    """
    v = _as_array(vol_forecast)
    # Bar volatilitesini tutma ufkuna, sonra yıllığa taşı.
    horizon_vol = v * np.sqrt(max(holding_bars, 1))
    ann = horizon_vol * np.sqrt(bars_per_year / max(holding_bars, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = np.where(ann > 0, target_annual_vol / ann, 0.0)
    lev = np.nan_to_num(lev, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(lev, 0.0, float(max_leverage))


def compute_position_size(
    prob: ArrayLike,
    side: ArrayLike,
    vol_forecast: ArrayLike | None = None,
    payoff_ratio: float | None = None,
    bars_per_year: float = 525_600.0,
    holding_bars: int = 60,
    cfg: SizingConfig | None = None,
) -> np.ndarray:
    """Nihai pozisyon boyutunu (yönlü) hesaplar.

    Args:
        prob: Meta-modelin "işlem kârlı" olasılığı.
        side: İşlem yönü (+1 / -1), birincil modelden gelir.
        vol_forecast: Bar başına volatilite tahmini (vol targeting için).
        payoff_ratio: Kelly için kazanç/kayıp oranı. ``None`` ise config veya
            1.0 kullanılır.
        bars_per_year: Yıllıklaştırma katsayısı.
        holding_bars: Tipik tutma süresi.
        cfg: Boyutlandırma konfigürasyonu.

    Returns:
        Yönlü pozisyon boyutu dizisi. ``|boyut| < min_position`` olanlar 0
        yapılır (işleme girilmez); ``max_position`` ile kırpılır.

    Raises:
        ValueError: Bilinmeyen ``cfg.method`` verilirse.
    """
    c = cfg or CONFIG.sizing
    p = _as_array(prob)
    s = _as_array(side)
    b = payoff_ratio if payoff_ratio is not None else (c.payoff_ratio or 1.0)

    prob_sz = prob_to_bet_size(p, method="zscore", scale=c.prob_scale)
    kelly_sz = kelly_size(p, payoff_ratio=b, fraction=c.kelly_fraction, cap=c.kelly_cap)
    if vol_forecast is None:
        vol_sz = np.ones_like(p)
    else:
        vol_sz = vol_target_size(
            vol_forecast,
            target_annual_vol=c.target_annual_vol,
            bars_per_year=bars_per_year,
            holding_bars=holding_bars,
            max_leverage=c.max_leverage,
        )

    if c.method == "prob":
        size = prob_sz
    elif c.method == "kelly":
        size = kelly_sz
    elif c.method == "vol_target":
        size = vol_sz
    elif c.method == "combined":
        # Olasılık boyutu ile Kelly'nin KÜÇÜĞÜ alınır (muhafazakâr),
        # sonra volatilite hedeflemesiyle ölçeklenir.
        size = np.minimum(prob_sz, np.maximum(kelly_sz, 0.0)) * vol_sz
    else:
        raise ValueError(f"Bilinmeyen sizing method: {c.method}")

    size = np.clip(size, 0.0, c.max_position)
    size = np.where(size < c.min_position, 0.0, size)
    return size * np.sign(s)


def payoff_ratio_from_barriers(pt_mult: float, sl_mult: float) -> float:
    """Bariyer çarpanlarından Kelly için kazanç/kayıp oranını türetir.

    Args:
        pt_mult: Kâr-al bariyeri çarpanı.
        sl_mult: Zarar-kes bariyeri çarpanı.

    Returns:
        ``b = pt / sl`` oranı.

    Notes:
        Bu yalnızca bir ÜST TAHMİNDİR: gerçek kazanç/kayıp oranı, süre
        bariyeriyle kapanan işlemler ve maliyetler yüzünden daha düşüktür.
        Gerçekleşmiş oranı backtest çıktısından almak daha doğrudur.
    """
    return float(pt_mult) / max(float(sl_mult), 1e-9)


def discretize_size(size: ArrayLike, step: float = 0.05) -> np.ndarray:
    """Pozisyon boyutunu basamaklara yuvarlar (gereksiz işlem trafiğini azaltır).

    Args:
        size: Sürekli pozisyon boyutu.
        step: Basamak genişliği.

    Returns:
        Basamaklandırılmış boyutlar.

    Notes:
        Sürekli boyut, her barda küçük ayarlamalar yaparak komisyonu şişirir.
        Basamaklandırma bu "boyut gürültüsünü" keser.
    """
    s = _as_array(size)
    if step <= 0:
        return s
    return np.round(s / step) * step
