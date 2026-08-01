"""Çapraz doğrulama ve istatistiksel anlamlılık.

Finansal veride standart K-Fold **YANLIŞTIR**. İki sebepten:

1. **Etiket çakışması.** Bir etiket ``t``'den ``t1``'e kadar bilgi taşır. Test
   kümesindeki bir olayla eğitim kümesindeki bir olayın ömürleri kesişiyorsa,
   model test dönemine ait bilgiyi eğitimde görmüş olur. Çözüm: **purging**.
2. **Seri korelasyon.** Kesişim olmasa bile, test kümesinin hemen ardından
   gelen eğitim örnekleri hâlâ ilişkilidir. Çözüm: **embargo**.

Ayrıca tek bir CV skoru yanıltıcıdır: bir kez şanslı bir bölünme yakalamak
kolaydır. Bu yüzden Combinatorial Purged CV ile bir performans DAĞILIMI
üretiyor, ve nihai olarak Deflated Sharpe Ratio ile "kaç deneme yaptık?"
sorusunu hesaba katıyoruz.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import BaseCrossValidator

from .config import CONFIG, CVConfig


# --------------------------------------------------------------------------- #
# Purged K-Fold + Embargo
# --------------------------------------------------------------------------- #


def _purge_train_indices(
    t1: pd.Series,
    test_positions: np.ndarray,
    embargo_bars: int,
) -> np.ndarray:
    """Test kümesiyle çakışan ve embargo bölgesine düşen eğitim örneklerini eler.

    Args:
        t1: Olay -> etiket bitiş zamanı eşlemesi (olay zamanına göre SIRALI).
        test_positions: Test örneklerinin konum indeksleri.
        embargo_bars: Test kümesinin sonrasında kaç ÖRNEK karantinaya alınacak.

    Returns:
        Kullanılabilir eğitim örneklerinin konum indeksleri.

    Notes:
        Purging iki yönlüdür:

        * **Geriye doğru:** ``t1_train >= t0_test`` olan eğitim örnekleri, test
          dönemine uzanan etikete sahiptir -> atılır.
        * **İleriye doğru:** ``t0_train <= t1_test`` olan eğitim örnekleri, test
          etiketinin ömrü içinde başlamıştır -> atılır.
        * **Embargo:** test bloğunun hemen ardındaki örnekler seri korelasyon
          nedeniyle atılır.
    """
    n = len(t1)
    t0_all = t1.index
    t1_all = pd.DatetimeIndex(t1.to_numpy())

    test_start = t0_all[test_positions.min()]
    test_end = t1_all[test_positions].max()

    mask = np.ones(n, dtype=bool)
    mask[test_positions] = False

    overlap = (t1_all >= test_start) & (t0_all <= test_end)
    mask &= ~overlap

    if embargo_bars > 0:
        hi = min(n, int(test_positions.max()) + 1 + embargo_bars)
        mask[int(test_positions.max()) + 1 : hi] = False

    return np.flatnonzero(mask)


class PurgedKFold(BaseCrossValidator):
    """Purging + embargo uygulayan K-Fold (López de Prado, AFML Böl. 7).

    Test blokları zaman sırasına göre BİTİŞİK seçilir (karıştırma yoktur).

    Attributes:
        n_splits: Katman sayısı.
        t1: Olay -> etiket bitiş zamanı eşlemesi.
        embargo_pct: Toplam örneğin yüzdesi olarak embargo genişliği.
    """

    def __init__(
        self,
        t1: pd.Series,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        """Kurucu.

        Args:
            t1: Olay zamanından etiket bitiş zamanına eşleme (artan sıralı).
            n_splits: Katman sayısı.
            embargo_pct: Embargo oranı.

        Raises:
            ValueError: ``t1`` sıralı değilse veya ``n_splits < 2`` ise.
        """
        if n_splits < 2:
            raise ValueError("n_splits en az 2 olmalı.")
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 indeksi artan sıralı olmalı.")
        self.n_splits = int(n_splits)
        self.t1 = t1
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D102, N803
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame | np.ndarray | None = None,  # noqa: N803
        y: pd.Series | None = None,
        groups: Sequence | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Eğitim/test konum indekslerini üretir.

        Args:
            X: Özellik matrisi (indeksi ``t1`` ile aynı olmalı).
            y: Kullanılmaz (sklearn uyumu için).
            groups: Kullanılmaz.

        Yields:
            ``(train_positions, test_positions)`` ikilileri.

        Raises:
            ValueError: ``X`` indeksi ``t1`` ile uyuşmuyorsa.
        """
        n = len(self.t1)
        if X is not None and hasattr(X, "index") and len(X) != n:
            raise ValueError("X ile t1 uzunlukları uyuşmuyor.")

        embargo_bars = int(n * self.embargo_pct)
        bounds = np.array_split(np.arange(n), self.n_splits)
        for test_positions in bounds:
            if len(test_positions) == 0:
                continue
            train_positions = _purge_train_indices(self.t1, test_positions, embargo_bars)
            yield train_positions, test_positions


# --------------------------------------------------------------------------- #
# Combinatorial Purged CV
# --------------------------------------------------------------------------- #


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (AFML Böl. 12).

    Veriyi ``N`` bitişik gruba böler ve her denemede ``k`` grubu test olarak
    seçer -> ``C(N, k)`` farklı bölünme. Her bölünmede purging + embargo
    uygulanır.

    Neden? Tek bir walk-forward yolu, tek bir tarih dizilimidir; "şanslı yol"
    riski yüksektir. CPCV, ``C(N,k) * k / N`` adet farklı BACKTEST YOLU üretir
    ve böylece tek bir skor yerine bir DAĞILIM verir. Overfitting olasılığını
    (PBO) değerlendirmenin de temelidir.

    Attributes:
        t1: Olay -> etiket bitiş zamanı.
        n_groups: Toplam grup sayısı (N).
        n_test_groups: Test grubu sayısı (k).
        embargo_pct: Embargo oranı.
    """

    def __init__(
        self,
        t1: pd.Series,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        """Kurucu.

        Args:
            t1: Olay -> etiket bitiş zamanı eşlemesi.
            n_groups: Grup sayısı.
            n_test_groups: Her denemede test edilecek grup sayısı.
            embargo_pct: Embargo oranı.

        Raises:
            ValueError: ``n_test_groups >= n_groups`` ise.
        """
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups < n_groups olmalı.")
        self.t1 = t1
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.embargo_pct = float(embargo_pct)

    @property
    def n_splits(self) -> int:
        """Toplam bölünme (deneme) sayısı: ``C(N, k)``."""
        return len(list(combinations(range(self.n_groups), self.n_test_groups)))

    @property
    def n_paths(self) -> int:
        """Üretilebilecek bağımsız backtest yolu sayısı: ``C(N,k) * k / N``."""
        return int(self.n_splits * self.n_test_groups / self.n_groups)

    def split(
        self,
        X: pd.DataFrame | None = None,  # noqa: N803
    ) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
        """Tüm grup kombinasyonları için bölünmeleri üretir.

        Args:
            X: Özellik matrisi (uzunluk kontrolü için, opsiyonel).

        Yields:
            ``(train_positions, test_positions, test_group_ids)`` üçlüleri.
        """
        n = len(self.t1)
        embargo_bars = int(n * self.embargo_pct)
        groups = np.array_split(np.arange(n), self.n_groups)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_positions = np.sort(np.concatenate([groups[g] for g in combo]))
            # Purging'i her bitişik test bloğu için ayrı ayrı uygulayıp
            # kesişimini almak gerekir: tek blok varsayımı yanlış sonuç verir.
            mask = np.ones(n, dtype=bool)
            for g in combo:
                block = groups[g]
                allowed = _purge_train_indices(self.t1, block, embargo_bars)
                blocked = np.ones(n, dtype=bool)
                blocked[allowed] = False
                mask &= ~blocked
            train_positions = np.flatnonzero(mask)
            yield train_positions, test_positions, combo


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = False,
    t1: pd.Series | None = None,
    embargo_pct: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward (ileri yürüyüş) bölünmeleri üretir.

    Gerçek dağıtımın (deployment) en yakın taklidi: geçmişte eğit, hemen
    sonrasında test et, pencereyi kaydır.

    Args:
        n_samples: Toplam örnek sayısı.
        train_size: Eğitim penceresi uzunluğu.
        test_size: Test penceresi uzunluğu.
        step: Kaydırma adımı. ``None`` ise ``test_size``.
        expanding: ``True`` ise eğitim penceresi genişler, ``False`` ise kayar.
        t1: Verilirse purging + embargo da uygulanır.
        embargo_pct: Embargo oranı (yalnızca ``t1`` verilirse).

    Returns:
        ``(train_positions, test_positions)`` listesi.
    """
    st = step or test_size
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while start + train_size + test_size <= n_samples:
        train_lo = 0 if expanding else start
        train_hi = start + train_size
        train_idx = np.arange(train_lo, train_hi)
        test_idx = np.arange(train_hi, train_hi + test_size)
        if t1 is not None:
            embargo_bars = int(len(t1) * embargo_pct)
            allowed = _purge_train_indices(t1, test_idx, embargo_bars)
            train_idx = np.intersect1d(train_idx, allowed, assume_unique=False)
        if len(train_idx) > 0:
            splits.append((train_idx, test_idx))
        start += st
    return splits


# --------------------------------------------------------------------------- #
# Sharpe ve türevleri
# --------------------------------------------------------------------------- #


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: float = 1.0,
    risk_free: float = 0.0,
) -> float:
    """Yıllıklaştırılmış Sharpe oranı.

    Args:
        returns: Periyot getirileri (oran).
        periods_per_year: Yıllıklaştırma katsayısı.
        risk_free: Periyot başına risksiz getiri.

    Returns:
        Yıllıklaştırılmış Sharpe. Std sıfırsa 0.
    """
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    excess = r - risk_free
    sd = excess.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    "Gözlenen Sharpe'ın gerçekten ``benchmark_sr``'den büyük olma olasılığı",
    örneklem uzunluğunu ve getirilerin çarpıklık/basıklığını hesaba katarak.

    Args:
        observed_sr: Gözlenen Sharpe (YILLIKLAŞTIRILMAMIŞ, periyot bazında).
        benchmark_sr: Karşılaştırma eşiği (aynı birimde).
        n_obs: Gözlem sayısı.
        skew: Getirilerin çarpıklığı.
        kurtosis: Getirilerin basıklığı (normal = 3).

    Returns:
        0-1 arası olasılık.
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    if denom <= 0:
        return 0.0
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Şans eseri elde edilebilecek BEKLENEN EN YÜKSEK Sharpe.

    ``n_trials`` bağımsız denemenin hepsi gerçekte sıfır Sharpe'a sahip olsa
    bile, en iyisi pozitif çıkar. Bu fonksiyon o "şans eşiğini" verir.

    Args:
        n_trials: Denenen strateji/konfigürasyon sayısı.
        sr_variance: Denemeler arası Sharpe varyansı.

    Returns:
        Beklenen maksimum Sharpe (aynı birimde).
    """
    n = max(int(n_trials), 2)
    gamma = 0.5772156649015329  # Euler-Mascheroni
    e = np.e
    emc = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n) + gamma * norm.ppf(1.0 - 1.0 / (n * e))
    return float(np.sqrt(max(sr_variance, 0.0)) * emc)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (DSR) — Bailey & López de Prado.

    DSR, gözlenen Sharpe'ı **çoklu deneme (multiple testing)** düzeltmesiyle
    değerlendirir: karşılaştırma eşiği 0 değil, ``n_trials`` deneme sonucunda
    şans eseri beklenen maksimum Sharpe'tır.

    DÜRÜSTLÜK UYARISI: ``n_trials``, hattı çalıştırırken denediğiniz TÜM
    varyasyonları (özellik seti, hiperparametre, bariyer, eşik...) içermelidir.
    Küçük beyan etmek DSR'yi yapay olarak şişirir ve tüm testi anlamsızlaştırır.

    Args:
        observed_sr: Gözlenen Sharpe (periyot bazında, yıllıklaştırılmamış).
        n_obs: Gözlem sayısı.
        n_trials: Denenen strateji sayısı.
        sr_variance: Denemelerin Sharpe'larının varyansı.
        skew: Getiri çarpıklığı.
        kurtosis: Getiri basıklığı.

    Returns:
        ``dsr`` (0-1 olasılık), ``sr0`` (şans eşiği), ``psr_vs_zero`` sözlüğü.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    dsr = probabilistic_sharpe_ratio(observed_sr, sr0, n_obs, skew, kurtosis)
    psr0 = probabilistic_sharpe_ratio(observed_sr, 0.0, n_obs, skew, kurtosis)
    return {
        "dsr": float(dsr),
        "sr0_sans_esigi": float(sr0),
        "psr_vs_zero": float(psr0),
        "n_trials": float(n_trials),
        "n_obs": float(n_obs),
    }


def deflated_sharpe_from_returns(
    returns: pd.Series | np.ndarray,
    trial_sharpes: Sequence[float] | np.ndarray | None = None,
    n_trials: int | None = None,
    periods_per_year: float = 1.0,
    trial_sharpes_annualized: bool = True,
    cfg: CVConfig | None = None,
) -> dict[str, float]:
    """Getiri serisinden doğrudan DSR hesaplar (pratik sarmalayıcı).

    Args:
        returns: Strateji periyot getirileri (``periods_per_year`` ile aynı
            frekansta olmalı).
        trial_sharpes: Denenen tüm konfigürasyonların Sharpe'ları. Verilirse
            ``sr_variance`` bunlardan, ``n_trials`` da uzunluğundan türetilir.
        n_trials: Deneme sayısını elle vermek için.
        periods_per_year: Yıllıklaştırma katsayısı.
        trial_sharpes_annualized: ``trial_sharpes`` YILLIKLAŞTIRILMIŞ mı?
            BİRİM UYUMU KRİTİKTİR: gözlenen Sharpe periyot bazında hesaplanır;
            deneme Sharpe'ları yıllık verilirse ``sqrt(periods_per_year)`` ile
            geri ölçeklenmeleri gerekir. Aksi halde şans eşiği (``sr0``) yüzlerce
            kat büyük çıkar ve DSR daima 0 görünür.
        cfg: CV konfigürasyonu (varsayılan ``n_trials`` kaynağı).

    Returns:
        DSR sözlüğü + yıllıklaştırılmış Sharpe.
    """
    c = cfg or CONFIG.cv
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {"dsr": 0.0, "sr0_sans_esigi": 0.0, "psr_vs_zero": 0.0,
                "sharpe_yillik": 0.0, "n_trials": 0.0, "n_obs": float(len(r))}

    sd = r.std(ddof=1)
    sr_per_period = float(r.mean() / sd) if sd > 0 else 0.0
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt() + 3.0)  # pandas fazlalık basıklık döndürür

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        ts = np.asarray(trial_sharpes, dtype="float64")
        ts = ts[np.isfinite(ts)]
        if trial_sharpes_annualized and periods_per_year > 0:
            # Periyot bazına indir: gözlenen Sharpe ile aynı birim olmalı.
            ts = ts / np.sqrt(periods_per_year)
        sr_var = float(ts.var(ddof=1)) if len(ts) > 1 else 0.01
        trials = int(n_trials) if n_trials is not None else max(len(ts), c.n_trials)
    else:
        # Deneme dağılımı yoksa muhafazakâr bir varsayım kullanılır.
        sr_var = float(max(sr_per_period**2, 1e-6))
        trials = int(n_trials) if n_trials is not None else c.n_trials

    out = deflated_sharpe_ratio(sr_per_period, len(r), trials, sr_var, skew, kurt)
    out["sharpe_yillik"] = sr_per_period * float(np.sqrt(periods_per_year))
    out["sr_varyans_denemeler"] = sr_var
    return out


# --------------------------------------------------------------------------- #
# Skor dağılımı özeti
# --------------------------------------------------------------------------- #


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """CV skor dağılımını özetler (ortalama tek başına yeterli değildir).

    Args:
        scores: Her satırı bir bölünme olan skor tablosu.

    Returns:
        Metrik başına ortalama/std/min/medyan/maks/pozitif-oran tablosu.
    """
    if scores.empty:
        return pd.DataFrame()
    num = scores.select_dtypes(include=[np.number])
    out = pd.DataFrame(
        {
            "ortalama": num.mean(),
            "std": num.std(ddof=1),
            "min": num.min(),
            "medyan": num.median(),
            "maks": num.max(),
            "pozitif_oran": (num > 0).mean(),
        }
    )
    return out.round(6)
