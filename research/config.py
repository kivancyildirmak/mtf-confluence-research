"""Araştırma hattının TEK merkezi konfigürasyon dosyası.

Buradaki ilke: hattın hiçbir yerinde "sihirli sayı" (magic number) bulunmayacak.
Tüm parametreler bu modüldeki dataclass'lardan okunur. Böylece:

* Backtest ile ileride yazılacak CANLI bot birebir aynı parametrelerle çalışır
  (özellik pariteleri bozulmaz).
* Tekrarlanabilirlik: tek bir ``RANDOM_SEED`` tüm kütüphanelere dağıtılır.
* Bir deneyi tekrar üretmek için sadece bu dosyayı (veya ``ResearchConfig``
  örneğini) saklamak yeterlidir; ``model.save_model`` config'i modelle birlikte
  diske yazar.

Zaman birimi varsayımı: tüm "pencere" (window) parametreleri BAR cinsindendir.
Varsayılan bar periyodu 1 dakikadır (``DataConfig.bar_minutes``).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Tekrarlanabilirlik
# --------------------------------------------------------------------------- #

#: Hattın tamamında kullanılan tek rastgelelik tohumu.
RANDOM_SEED: int = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Tüm rastgelelik kaynaklarını sabitler.

    LightGBM kendi tohumunu :class:`ModelConfig` üzerinden alır; burada
    Python/NumPy tarafı sabitlenir.

    Args:
        seed: Kullanılacak tohum değeri.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Veri katmanı
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Veri kaynağı ve zaman ızgarası ayarları."""

    #: İşlem yapılacak ana sembol (Paribu formatı).
    symbol: str = "BTC_TL"
    #: Çapraz varlık (BTC öncülüğü) için referans sembol.
    reference_symbol: str = "BTC_USDT"
    #: TL primi hesabı için stablecoin paritesi (veri yoksa placeholder).
    stable_symbol: str = "USDT_TL"
    #: Bar periyodu (dakika). Tüm pencere parametreleri bu bara göre yorumlanır.
    bar_minutes: int = 1
    #: Ham verinin okunacağı klasör.
    data_dir: str = "data"
    #: Çıktıların (model, grafik, rapor) yazılacağı klasör.
    output_dir: str = "artifacts"
    #: Sentetik veri üretiminde kaç bar üretileceği (yaklaşık 45 gün, 1dk).
    synthetic_bars: int = 65_000
    #: Sentetik veri başlangıç zamanı (UTC).
    synthetic_start: str = "2025-01-01 00:00:00"
    #: Sentetik veride başlangıç fiyatı (TL).
    synthetic_start_price: float = 3_000_000.0
    #: Sentetik veride yıllık volatilite hedefi.
    synthetic_annual_vol: float = 0.65

    @property
    def bars_per_year(self) -> float:
        """Yıllıklaştırma katsayısı için yıl başına bar sayısı."""
        return 365.0 * 24.0 * 60.0 / float(self.bar_minutes)

    @property
    def bars_per_day(self) -> float:
        """Gün başına bar sayısı."""
        return 24.0 * 60.0 / float(self.bar_minutes)


# --------------------------------------------------------------------------- #
# Veri toplayıcı (poll-forward)
# --------------------------------------------------------------------------- #


@dataclass
class CollectorConfig:
    """Paribu public ticker toplayıcısının ayarları.

    Paribu'da tarihsel mum/işlem/defter ucu YOKTUR (probe ile doğrulandı:
    candles/orderbook/trades -> 404). Elimizdeki tek public uç anlık
    ticker'dır; bu yüzden geçmiş ancak BUGÜNDEN İTİBAREN biriktirilebilir.
    """

    #: Doğrulanmış public ticker ucu (anahtar/imza gerektirmez).
    ticker_url: str = "https://www.paribu.com/ticker"
    #: Toplanacak pariteler (ticker yanıtındaki anahtarlar).
    symbols: tuple[str, ...] = ("BTC_TL",)
    #: Yoklama aralığı (saniye). Dakika içi high/low'u makul yakalamak için
    #: sık tutulur; küçültmek doğruluğu artırır ama rate-limit riskini de.
    poll_interval_seconds: float = 5.0
    #: Ham tick'lerin yazılacağı klasör (gün başına bir parquet).
    tick_dir: str = "research/data/ticks"
    #: Toplulaştırılmış barların yazılacağı klasör.
    ohlcv_dir: str = "research/data/ohlcv"
    #: Bellekteki tampon kaç saniyede bir diske yazılsın. Ani kapanmada en
    #: fazla bu kadarlık tick kaybedilir.
    flush_interval_seconds: float = 30.0
    #: "Çalışıyorum" kalp atışı logu aralığı (saniye).
    heartbeat_interval_seconds: float = 60.0
    #: HTTP zaman aşımı (saniye).
    request_timeout: float = 15.0
    #: Hata sonrası ilk geri çekilme süresi (saniye).
    backoff_base_seconds: float = 2.0
    #: Üstel geri çekilmenin üst sınırı (saniye).
    backoff_max_seconds: float = 300.0
    #: İstek başlığı (borsaya kim olduğumuzu bildirmek nezakettir).
    user_agent: str = "paribu-research-collector/1.0"


# --------------------------------------------------------------------------- #
# Özellik mühendisliği
# --------------------------------------------------------------------------- #


@dataclass
class FeatureConfig:
    """``features.make_features`` tarafından kullanılan pencere uzunlukları.

    DİKKAT (feature parity): Bu değerler değişirse eğitilmiş model geçersiz olur.
    ``model.save_model`` config'i modelle beraber sakladığı için canlı bot
    yüklerken parametre uyuşmazlığını tespit edebilir.
    """

    #: Log-getiri ufukları (bar). 1/5/15/60 dakika (1dk barda).
    return_horizons: tuple[int, ...] = (1, 5, 15, 60)
    #: Kayan ortalama/std pencereleri (bar).
    ma_windows: tuple[int, ...] = (5, 15, 60, 240)
    #: Gerçekleşmiş volatilite pencereleri (bar).
    vol_windows: tuple[int, ...] = (15, 60, 240)
    #: ATR pencereleri (bar).
    atr_windows: tuple[int, ...] = (14, 60)
    #: Vol-of-vol için, volatilite serisinin std'sinin alındığı pencere.
    vol_of_vol_window: int = 240
    #: RSI pencereleri.
    rsi_windows: tuple[int, ...] = (14, 60)
    #: MACD (hızlı, yavaş, sinyal) parametreleri.
    macd_params: tuple[int, int, int] = (12, 26, 9)
    #: ROC (Rate of Change) pencereleri.
    roc_windows: tuple[int, ...] = (15, 60)
    #: Fiyat z-skoru pencereleri.
    zscore_windows: tuple[int, ...] = (60, 240)
    #: Hacim z-skoru pencereleri.
    volume_windows: tuple[int, ...] = (15, 60, 240)
    #: Kayan VWAP pencereleri.
    vwap_windows: tuple[int, ...] = (60, 240)
    #: Çapraz varlık korelasyon pencereleri.
    corr_windows: tuple[int, ...] = (60, 240)
    #: BTC öncülüğü için test edilecek gecikmeler (bar).
    lead_lag_bars: tuple[int, ...] = (1, 5, 15)
    #: Kayan skew/kurtosis pencereleri.
    moment_windows: tuple[int, ...] = (60, 240)
    #: Volatilite rejimi bayrağı için referans (uzun) pencere.
    regime_window: int = 1_440
    #: Rejim bayrağının "yüksek vol" eşiği (uzun dönem medyanın katı).
    regime_high_mult: float = 1.5
    #: Sonsuz/aşırı değerleri kırpma sınırı (winsorize benzeri güvenlik).
    clip_abs: float = 1e6


# --------------------------------------------------------------------------- #
# Etiketleme
# --------------------------------------------------------------------------- #


@dataclass
class LabelConfig:
    """Triple-barrier ve meta-labeling ayarları (López de Prado, AFML Böl. 3)."""

    #: Hedef volatilite tahmini için EWM span'ı (bar).
    vol_span: int = 240
    #: Bariyer çarpanları: (kâr-al, zarar-kes). Hedef vol ile ölçeklenir.
    #: KRİTİK: bariyerler gidiş-dönüş maliyete (~%0.5) göre yeterince GENİŞ
    #: olmalı. Dar bariyer + yüksek komisyon = matematiksel olarak kaybeden
    #: strateji; model ne kadar iyi olursa olsun kurtaramaz.
    pt_sl: tuple[float, float] = (2.0, 2.0)
    #: Dikey (süre) bariyer: kaç bar sonra pozisyon zorla kapanır (4 saat).
    vertical_bars: int = 240
    #: CUSUM olay filtresi eşiği (hedef vol'ün katı). Örnek çakışmasını azaltır.
    #: Düşürmek olay sayısını artırır ama etiket çakışmasını da büyütür.
    cusum_threshold_mult: float = 0.35
    #: Bu değerin altındaki beklenen getirili olaylar elenir (gürültü filtresi).
    min_ret: float = 0.0
    #: Bariyer teması bar içi high/low ile mi ölçülsün? (True = daha gerçekçi)
    use_intrabar_extremes: bool = True
    #: Aynı barda hem TP hem SL değerse hangisi kabul edilsin (kötümser = "sl").
    tie_break: str = "sl"
    #: Örnek ağırlıklarında zaman sönümü (LdP time decay). 1.0 = sönüm yok,
    #: 0.0 = en eski örnek sıfır ağırlık alır.
    time_decay_last_weight: float = 0.5
    #: Ağırlıklar getiri büyüklüğüne göre de ölçeklensin mi?
    weight_by_return: bool = True


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """LightGBM hiperparametreleri ve SHAP ayarları."""

    #: LightGBM ham parametreleri (``lgb.train`` sözlüğü).
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
            "bagging_seed": RANDOM_SEED,
            "data_random_seed": RANDOM_SEED,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 0,
        }
    )
    #: Maksimum ağaç sayısı.
    num_boost_round: int = 400
    #: Erken durdurma sabrı (doğrulama seti verildiğinde).
    early_stopping_rounds: int = 50
    #: SHAP hesabında kullanılacak maksimum örnek sayısı (hız için).
    shap_max_samples: int = 3_000
    #: Grafikte gösterilecek en önemli özellik sayısı.
    shap_top_n: int = 25


# --------------------------------------------------------------------------- #
# Çapraz doğrulama
# --------------------------------------------------------------------------- #


@dataclass
class CVConfig:
    """Purged K-Fold, Combinatorial Purged CV ve walk-forward ayarları."""

    #: Purged K-Fold katman sayısı.
    n_splits: int = 5
    #: Embargo oranı (toplam örnek sayısının yüzdesi olarak).
    embargo_pct: float = 0.01
    #: Combinatorial Purged CV: toplam grup sayısı (N).
    cpcv_n_groups: int = 6
    #: Combinatorial Purged CV: her denemede test edilecek grup sayısı (k).
    cpcv_n_test_groups: int = 2
    #: Walk-forward: eğitim penceresi (olay sayısı).
    wf_train_size: int = 2_000
    #: Walk-forward: test penceresi (olay sayısı).
    wf_test_size: int = 500
    #: Walk-forward: pencere kaydırma adımı (olay sayısı).
    wf_step: int = 500
    #: Walk-forward eğitim penceresi sabit mi (rolling) yoksa genişleyen mi?
    wf_expanding: bool = False
    #: Deflated Sharpe hesabında beyan edilen deneme sayısı (N trials).
    #:
    #: DÜRÜSTLÜK NOTU: burayı küçük tutmak DSR'yi yapay olarak ŞİŞİRİR. Sayı,
    #: yalnızca "kaç kez çalıştırdım"ı değil, araştırmacının kullandığı TÜM
    #: serbestlik derecelerini yansıtmalıdır (bariyerler, eşikler, boyutlandırma,
    #: özellik grupları, bar boyutu...). Defter README bölüm 11'de tutulur.
    #:
    #: Mevcut değer, uçtan uca değerlendirilen ~8 konfigürasyon × ~15 belgelenmiş
    #: serbestlik derecesinden gelir. Bilinçli olarak YUKARI yuvarlanmıştır:
    #: fazla beyan etmek testi yalnızca sertleştirir, az beyan etmek ise
    #: sonucu sahte biçimde anlamlı gösterir.
    n_trials: int = 120


# --------------------------------------------------------------------------- #
# Backtest / maliyetler
# --------------------------------------------------------------------------- #


@dataclass
class BacktestConfig:
    """Maliyet ve icra (execution) varsayımları.

    Bu sayılar hattın en kritik varsayımlarıdır: maliyetler hafife alınırsa
    araştırma sonucu tamamen yanıltıcı olur.
    """

    #: Tek yön komisyon (taker) — oran olarak. Paribu taker ~%0.20 varsayımı.
    commission_rate: float = 0.0020
    #: Tek yön slippage (baz puan, 1 bps = %0.01).
    slippage_bps: float = 5.0
    #: Emir gecikmesi (bar). Sinyal bar t kapanışında üretilir, emir t+latency
    #: barının AÇILIŞINDA gerçekleşir. Look-ahead'i engelleyen ana mekanizma.
    latency_bars: int = 1
    #: İşleme girmek için gereken minimum model olasılığı.
    prob_threshold: float = 0.55
    #: Beklenen getirinin toplam maliyetin kaç katı olması gerektiği.
    min_edge_over_cost: float = 1.0
    #: Aynı anda birden fazla pozisyon açılsın mı? (False = tek pozisyon)
    allow_overlapping: bool = False
    #: Başlangıç sermayesi (TL) — raporlama içindir.
    initial_capital: float = 100_000.0
    #: Yıllıklaştırmada kullanılacak risksiz getiri (metrik frekansı başına).
    risk_free_rate: float = 0.0
    #: Sharpe/Sortino/DSR'nin hesaplandığı frekans. 1 dakikalık barlarda
    #: doğrudan Sharpe hesaplamak, nakitte geçen barların sıfır getirisi
    #: yüzünden oranı şişirir; saatliğe indirgemek bunu düzeltir.
    metric_freq: str = "1h"


# --------------------------------------------------------------------------- #
# Pozisyon boyutlandırma
# --------------------------------------------------------------------------- #


@dataclass
class SizingConfig:
    """Olasılığa dayalı boyutlandırma + Kelly + volatilite hedefleme."""

    #: Boyutlandırma yöntemi: "prob" | "kelly" | "vol_target" | "combined".
    method: str = "combined"
    #: Tek işlemde kullanılabilecek maksimum sermaye oranı.
    max_position: float = 1.0
    #: Minimum anlamlı pozisyon (altındaki sinyaller işleme dönmez).
    #: "combined" yöntemde üç faktör ÇARPILDIĞI için boyutlar küçüktür;
    #: bu eşiği yüksek tutmak tüm işlemleri sessizce eler.
    min_position: float = 0.01
    #: Kelly kesri (fractional Kelly). 1.0 = tam Kelly (ÖNERİLMEZ).
    kelly_fraction: float = 0.5
    #: Kelly çıktısına uygulanan üst kapak.
    kelly_cap: float = 0.5
    #: Olasılık-boyut dönüşümünde kullanılan ölçek (p-0.5 çarpanı).
    prob_scale: float = 2.0
    #: Yıllık hedef volatilite (vol targeting). Kripto varlığın kendi vol'ü
    #: bunun çok üstünde olduğu için sonuç tipik olarak <1 kaldıraçtır.
    target_annual_vol: float = 0.30
    #: Volatilite hedeflemede uygulanan maksimum kaldıraç.
    max_leverage: float = 3.0
    #: Kelly'de kullanılacak kazanç/kayıp oranı (b). Bariyerlerden türetilir.
    payoff_ratio: float | None = None


# --------------------------------------------------------------------------- #
# Grid (ızgara) stratejisi
# --------------------------------------------------------------------------- #


@dataclass
class GridConfig:
    """Mekanik grid stratejisinin parametreleri.

    Grid bir TAHMİN modeli değildir; kural tabanlı bir stratejidir. Yine de aynı
    maliyet ve dürüstlük disiplinine tabidir: maliyetsiz grid her zaman kârlı
    görünür ve bu görüntü tamamen sahtedir.
    """

    #: Aralığın alt sınırı (fiyat). ``None`` ise veriden türetilir.
    lower_price: float | None = None
    #: Aralığın üst sınırı (fiyat). ``None`` ise veriden türetilir.
    upper_price: float | None = None
    #: Sınırlar veriden türetilirken ilk fiyatın etrafında ±bu oran kullanılır.
    auto_range_pct: float = 0.15
    #: Kademe (grid seviyesi) sayısı.
    n_levels: int = 20
    #: Toplam sermaye (TL).
    total_capital: float = 100_000.0
    #: Kademe aralığı: ``"geometric"`` (eşit yüzde) veya ``"linear"``.
    spacing: str = "geometric"
    #: Mod: ``"static"`` (aralık sabit) veya ``"trailing"`` (fiyat çıkınca kayar).
    mode: str = "static"
    #: Tek yön komisyon oranı (ML backtest'iyle AYNI varsayım).
    commission_rate: float = 0.0020
    #: Tek yön slippage (baz puan). Komisyonla birlikte gidiş-dönüş ~%0.5.
    slippage_bps: float = 5.0
    #: Rejim analizi pencere uzunluğu (pandas frekansı).
    regime_window: str = "7D"
    #: Rejim etiketlemesinde "yatay" sayılacak azami net değişim (mutlak oran).
    sideways_threshold: float = 0.02
    #: Aralığı ilk N günün fiyat aralığından türet (0 = kapalı).
    #:
    #: SIZINTI KARŞITI: Aralığı dönemin TAMAMINA bakarak seçmek klasik bir
    #: look-ahead hatasıdır — gerçekte grid'i kurarken geleceği bilemezsiniz.
    #: Bu ayar açıkken ilk ``range_warmup_days`` gün yalnızca GÖZLENİR (işlem
    #: yok), aralık o pencereden belirlenir ve işlem ondan SONRA başlar.
    range_warmup_days: float = 7.0
    #: Isınma penceresinden türetilen aralığa eklenecek pay (üstten ve alttan).
    range_pad_pct: float = 0.0
    #: Stop-loss: fiyat alt sınırın bu oran ALTINA inerse tüm pozisyon satılır
    #: ve grid durur. ``None`` = stop yok (düşen bıçağı yakalamaya devam eder).
    stop_loss_pct: float | None = None


# --------------------------------------------------------------------------- #
# Kök konfigürasyon
# --------------------------------------------------------------------------- #


@dataclass
class ResearchConfig:
    """Tüm alt konfigürasyonları toplayan kök nesne."""

    seed: int = RANDOM_SEED
    data: DataConfig = field(default_factory=DataConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    grid: GridConfig = field(default_factory=GridConfig)

    def to_dict(self) -> dict[str, Any]:
        """Config'i JSON'a yazılabilir sözlüğe çevirir (deney kaydı için)."""
        return asdict(self)


#: Modüller arası paylaşılan varsayılan konfigürasyon örneği.
CONFIG = ResearchConfig()


def scale_config_for_bars(cfg: ResearchConfig, new_bar_minutes: int) -> ResearchConfig:
    """Konfigürasyonu farklı bir bar boyutuna uyarlar (deney için).

    **Neyi ölçekler:** yalnızca DUVAR SAATİNE bağlı olan tutma ufkunu.
    ``vertical_bars`` bar sayısı cinsindendir ama anlamı süredir; bar boyutu
    değişince aynı süreyi verecek şekilde yeniden hesaplanır. Örneğin 1 dk'da
    240 bar = 4 saat; 15 dk'da bu 16 bar, 1 saatte 4 bar olur. Sabit sayı
    gömülmez, oran config'ten türetilir.

    **Neyi ölçeklemez ve neden:** özellik pencereleri ve ``vol_span`` BAR
    cinsinden bırakılır. Duvar saatine göre küçültülselerdi göstergeler
    bozulurdu — 1 saatlik barda MACD(12,26,9) → MACD(1,1,1), RSI(14) → RSI(1)
    olur ki ikisi de tanımsızdır. Bar cinsinde bırakmak her çözünürlüğe aynı
    İSTATİSTİKSEL geçmişi verir.

    Bunun bilinen bedeli (deneyin bilinçli kabul ettiği karışıklık): kaba
    barlarda özellikler duvar saati olarak daha geriye bakar. Yani karşılaştırma
    saf "çözünürlük" karşılaştırması değil, "çözünürlük + geriye bakış"
    birleşimidir. Sonuçları yorumlarken bu akılda tutulmalıdır.

    Hedef volatilitenin bu ölçeklemeyle tutarlı kaldığına dikkat: hedef vol
    ``bar_vol * sqrt(vertical_bars)`` olduğundan ve ``bar_vol`` bar boyutuyla
    ``sqrt`` oranında büyüdüğünden, bariyer genişliği her çözünürlükte AYNI
    duvar saati oynaklığına denk gelir. Yani bariyerler karşılaştırılabilir.

    Args:
        cfg: Kaynak konfigürasyon.
        new_bar_minutes: Hedef bar süresi (dakika).

    Returns:
        Yeni, ölçeklenmiş kopya (girdi değiştirilmez).

    Raises:
        ValueError: Hedef bar süresi pozitif değilse.
    """
    import copy

    if new_bar_minutes <= 0:
        raise ValueError("new_bar_minutes pozitif olmalı.")

    out = copy.deepcopy(cfg)
    ratio = float(new_bar_minutes) / float(cfg.data.bar_minutes)
    out.data.bar_minutes = int(new_bar_minutes)
    out.labeling.vertical_bars = max(1, int(round(cfg.labeling.vertical_bars / ratio)))
    return out
