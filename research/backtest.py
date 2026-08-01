"""Maliyetli backtest motoru.

Bu modülün tek amacı **iyimserliği kırmaktır**. Sinyalin AUC'si 0.60 olabilir
ama komisyon + slippage + gecikme sonrası strateji pekâlâ para kaybediyor
olabilir. Bu yüzden burada üç maliyet kalemi de zorunludur:

* **Komisyon** — her yönde ``commission_rate``.
* **Slippage** — her yönde ``slippage_bps``; fiyata yön aleyhine uygulanır.
* **Gecikme (latency)** — sinyal bar ``t`` KAPANIŞINDA üretilir, emir
  ``t + latency_bars`` barının AÇILIŞINDA gerçekleşir.

**İCRA VARSAYIMLARI (bilinçli olarak kötümser).**

1. Giriş, olay barının kapanış fiyatından DEĞİL, gecikme sonrası barın
   açılışından yapılır. Aradaki fark (gap) doğrudan zarar/kâr olarak yansır.
2. Bariyer teması bar KAPANIŞINDA fark edilir; çıkış bir sonraki barın
   açılışında gerçekleşir. Yani etiketleme aşamasındaki "tam bariyer
   fiyatından çıkış" varsayımı burada KULLANILMAZ — gerçekte stop emirleri
   nadiren tam fiyattan dolar.
3. Aynı anda tek pozisyon (varsayılan). Pozisyon açıkken gelen sinyaller
   atlanır; bu, gerçek sermaye kısıtını taklit eder.
4. Kısmi dolum, emir defteri etkisi (market impact) ve fon (funding) maliyeti
   MODELLENMEZ — bunlar sonuçları daha da kötüleştirir, yani raporlanan
   performans bir ÜST SINIRDIR.

SIZINTI NOTU: Backtest, olay barından sonraki hiçbir fiyatı karar için
kullanmaz. Bariyer taraması :mod:`labeling` tarafında yapılmıştır ve orada da
tarama olay barının BİR SONRASINDAN başlar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CONFIG,
    BacktestConfig,
    DataConfig,
    LabelConfig,
    SizingConfig,
)
from .sizing import compute_position_size, payoff_ratio_from_barriers


@dataclass
class BacktestResult:
    """Backtest çıktılarının tamamı.

    Attributes:
        trades: İşlem bazında detay tablo.
        equity: Bar bazında sermaye eğrisi (mark-to-market).
        bar_returns: Bar bazında strateji getirileri.
        period_returns: Metrik frekansına (varsayılan saatlik) indirgenmiş
            getiriler — Sharpe/Sortino/DSR bunlar üzerinden hesaplanır.
        position: Bar bazında yönlü pozisyon büyüklüğü.
        metrics: Özet metrikler.
        config: Kullanılan maliyet/icra varsayımları.
    """

    trades: pd.DataFrame
    equity: pd.Series
    bar_returns: pd.Series
    position: pd.Series
    metrics: dict[str, float]
    period_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    config: dict[str, Any] = field(default_factory=dict)


def periods_per_year(freq: str) -> float:
    """Verilen frekans için yıl başına periyot sayısı.

    Args:
        freq: pandas frekans dizgesi (örn. ``"1h"``, ``"1D"``).

    Returns:
        Yıllıklaştırma katsayısı.
    """
    delta = pd.Timedelta(freq)
    return float(pd.Timedelta(days=365) / delta)


def to_period_returns(equity: pd.Series, freq: str = "1h") -> pd.Series:
    """Sermaye eğrisini daha kaba bir frekansa indirger ve getiriye çevirir.

    NEDEN? 1 dakikalık barlarda Sharpe'ı ``sqrt(525600)`` ile yıllıklaştırmak
    sayıyı sistematik olarak ŞİŞİRİR: strateji zamanın çoğunda nakitte
    olduğundan getirilerin çoğu tam sıfırdır, bu da standart sapmayı yapay
    olarak küçültür. Saatlik (veya günlük) frekansa indirgemek bu artefaktı
    büyük ölçüde giderir ve sonucu sektör pratiğiyle karşılaştırılabilir kılar.

    Args:
        equity: Bar bazında sermaye eğrisi.
        freq: Hedef frekans.

    Returns:
        Periyot getirileri.
    """
    if equity.empty:
        return pd.Series(dtype="float64", name="period_return")
    eq = equity.resample(freq).last().ffill()
    return eq.pct_change().dropna().rename("period_return")


def total_cost_rate(cfg: BacktestConfig | None = None) -> float:
    """Gidiş-dönüş (round-trip) toplam maliyet oranı.

    Args:
        cfg: Backtest konfigürasyonu.

    Returns:
        Komisyon + slippage'ın iki yön toplamı (oran).
    """
    c = cfg or CONFIG.backtest
    return 2.0 * (c.commission_rate + c.slippage_bps / 10_000.0)


def expected_trade_return(
    prob: np.ndarray,
    target: np.ndarray,
    pt_mult: float,
    sl_mult: float,
) -> np.ndarray:
    """Bariyer geometrisine göre beklenen ham (maliyetsiz) getiri.

    ``E[r] = p * (pt_mult * hedef_vol) - (1 - p) * (sl_mult * hedef_vol)``

    Args:
        prob: İşlemin kârlı olma olasılığı.
        target: Olay bazında hedef volatilite.
        pt_mult: Kâr-al çarpanı.
        sl_mult: Zarar-kes çarpanı.

    Returns:
        Beklenen getiri dizisi.

    Notes:
        İYİMSER bir tahmindir: süre bariyeriyle kapanan işlemleri (getirisi
        genelde küçük) hesaba katmaz. Bu yüzden filtrede ``min_edge_over_cost``
        katsayısıyla ek bir güvenlik payı istenir.
    """
    p = np.clip(prob, 0.0, 1.0)
    return p * (pt_mult * target) - (1.0 - p) * (sl_mult * target)


def run_backtest(
    df: pd.DataFrame,
    events: pd.DataFrame,
    prob: pd.Series,
    cfg: BacktestConfig | None = None,
    sizing_cfg: SizingConfig | None = None,
    label_cfg: LabelConfig | None = None,
    data_cfg: DataConfig | None = None,
    vol_forecast: pd.Series | None = None,
) -> BacktestResult:
    """Maliyetli, gecikmeli backtest'i çalıştırır.

    İşlem açma koşulları (hepsi sağlanmalı):

    1. ``prob >= prob_threshold``,
    2. ``beklenen_getiri >= min_edge_over_cost * toplam_maliyet``,
    3. hesaplanan pozisyon boyutu ``min_position``'dan büyük,
    4. (varsayılan) o anda açık pozisyon yok.

    Args:
        df: OHLCV verisi.
        events: :func:`labeling.get_triple_barrier_labels` çıktısı
            (``t1``, ``side``, ``target`` sütunları zorunlu).
        prob: Olay indeksli model olasılıkları. **Örneklem-dışı (out-of-sample)
            olmalıdır** — örneklem-içi olasılıkla yapılan backtest anlamsızdır.
        cfg: Maliyet/icra konfigürasyonu.
        sizing_cfg: Boyutlandırma konfigürasyonu.
        label_cfg: Bariyer çarpanlarının okunacağı konfigürasyon.
        data_cfg: Yıllıklaştırma için bar periyodu bilgisi.
        vol_forecast: Bar bazında volatilite tahmini (volatilite hedeflemesi
            için). ``None`` ise ``events["target"]`` ufuk volatilitesinden
            türetilir.

    Returns:
        :class:`BacktestResult`.

    Raises:
        ValueError: Zorunlu sütunlar eksikse.
    """
    c = cfg or CONFIG.backtest
    sc = sizing_cfg or CONFIG.sizing
    lc = label_cfg or CONFIG.labeling
    dc = data_cfg or CONFIG.data

    required = {"t1", "side", "target"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events tablosunda eksik sütunlar: {sorted(missing)}")

    n = len(df)
    idx = df.index
    open_px = df["open"].to_numpy(dtype="float64")
    close_px = df["close"].to_numpy(dtype="float64")

    ev = events.loc[events.index.isin(prob.index)].copy()
    ev = ev.sort_index()
    if ev.empty:
        return _empty_result(df, c)

    p = prob.reindex(ev.index).to_numpy(dtype="float64")
    ev_pos = idx.get_indexer(ev.index)
    touch_pos = idx.get_indexer(pd.DatetimeIndex(ev["t1"].to_numpy()))
    side = ev["side"].to_numpy(dtype="float64")
    target = ev["target"].to_numpy(dtype="float64")

    pt_mult, sl_mult = lc.pt_sl
    cost_rt = total_cost_rate(c)
    slip = c.slippage_bps / 10_000.0

    exp_ret = expected_trade_return(p, target, pt_mult, sl_mult)

    # Volatilite hedeflemesi için BAR başına vol tahmini (ufuk vol'ünden geri
    # ölçekleme). SIZINTI YOK: target zaten kayan pencereden gelir.
    if vol_forecast is not None:
        bar_vol = vol_forecast.reindex(ev.index).to_numpy(dtype="float64")
    else:
        bar_vol = target / np.sqrt(max(lc.vertical_bars, 1))

    payoff = (
        sc.payoff_ratio
        if sc.payoff_ratio is not None
        else payoff_ratio_from_barriers(pt_mult, sl_mult)
    )
    size = compute_position_size(
        p,
        side,
        vol_forecast=bar_vol,
        payoff_ratio=payoff,
        bars_per_year=dc.bars_per_year,
        holding_bars=lc.vertical_bars,
        cfg=sc,
    )

    # --- İşlem seçimi ------------------------------------------------------- #
    lat = int(c.latency_bars)
    rows: list[dict[str, Any]] = []
    last_exit_pos = -1

    for k in range(len(ev)):
        if p[k] < c.prob_threshold:
            continue
        if exp_ret[k] < c.min_edge_over_cost * cost_rt:
            continue
        w = float(size[k])
        if w == 0.0 or not np.isfinite(w):
            continue

        entry_pos = ev_pos[k] + lat
        exit_pos = min(int(touch_pos[k]) + lat, n - 1)
        if entry_pos >= n or entry_pos >= exit_pos:
            # Gecikme, pozisyonun ömrünü yutuyor -> işlem yapılamaz.
            continue
        if not c.allow_overlapping and entry_pos <= last_exit_pos:
            continue

        s = float(np.sign(w))
        # Slippage fiyata yön ALEYHİNE uygulanır (alışta yukarı, satışta aşağı).
        entry_price = open_px[entry_pos] * (1.0 + s * slip)
        exit_price = open_px[exit_pos] * (1.0 - s * slip)
        gross = s * (exit_price / entry_price - 1.0)
        commission = 2.0 * c.commission_rate
        net = abs(w) * (gross - commission)

        rows.append(
            {
                "event_time": ev.index[k],
                "entry_time": idx[entry_pos],
                "exit_time": idx[exit_pos],
                "entry_pos": entry_pos,
                "exit_pos": exit_pos,
                "side": s,
                "size": abs(w),
                "prob": p[k],
                "beklenen_getiri": exp_ret[k],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "brut_getiri": gross,
                "net_getiri": net,
                "maliyet": abs(w) * (commission + 2.0 * slip),
                "tutma_bar": exit_pos - entry_pos,
                "barrier": int(ev["barrier"].iloc[k]) if "barrier" in ev else 0,
            }
        )
        last_exit_pos = exit_pos

    trades = pd.DataFrame(rows)
    if trades.empty:
        return _empty_result(df, c)

    # --- Bar bazında mark-to-market sermaye eğrisi -------------------------- #
    equity = np.full(n, c.initial_capital, dtype="float64")
    position = np.zeros(n, dtype="float64")
    realized = c.initial_capital
    cursor = 0

    for tr in trades.itertuples(index=False):
        e, x = int(tr.entry_pos), int(tr.exit_pos)
        equity[cursor:e] = realized
        # Açık pozisyon süresince gerçekleşmemiş K/Z (slippage girişte,
        # komisyonun yarısı girişte düşülür; kalanı çıkışta gerçekleşir).
        mtm = tr.side * tr.size * (close_px[e:x] / tr.entry_price - 1.0)
        mtm = mtm - tr.size * c.commission_rate
        equity[e:x] = realized * (1.0 + mtm)
        position[e:x] = tr.side * tr.size
        realized = realized * (1.0 + tr.net_getiri)
        equity[x] = realized
        cursor = x + 1

    equity[cursor:] = realized

    equity_s = pd.Series(equity, index=idx, name="equity")
    bar_ret = equity_s.pct_change().fillna(0.0).rename("bar_return")
    position_s = pd.Series(position, index=idx, name="position")

    metrics = compute_metrics(trades, equity_s, bar_ret, c, dc)
    return BacktestResult(
        trades=trades,
        equity=equity_s,
        bar_returns=bar_ret,
        period_returns=to_period_returns(equity_s, c.metric_freq),
        position=position_s,
        metrics=metrics,
        config={
            "commission_rate": c.commission_rate,
            "slippage_bps": c.slippage_bps,
            "latency_bars": c.latency_bars,
            "prob_threshold": c.prob_threshold,
            "min_edge_over_cost": c.min_edge_over_cost,
            "toplam_maliyet_orani": cost_rt,
        },
    )


def _empty_result(df: pd.DataFrame, c: BacktestConfig) -> BacktestResult:
    """Hiç işlem açılmadığında dönen nötr sonuç."""
    equity = pd.Series(float(c.initial_capital), index=df.index, name="equity")
    zeros = pd.Series(0.0, index=df.index)
    return BacktestResult(
        trades=pd.DataFrame(),
        equity=equity,
        bar_returns=zeros.rename("bar_return"),
        period_returns=to_period_returns(equity, c.metric_freq),
        position=zeros.rename("position"),
        metrics={"islem_sayisi": 0.0, "toplam_getiri": 0.0, "sharpe": 0.0,
                 "max_drawdown": 0.0, "isabet_orani": float("nan")},
    )


# --------------------------------------------------------------------------- #
# Metrikler
# --------------------------------------------------------------------------- #


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, int]:
    """Maksimum düşüş (drawdown), dip zamanı ve süresi.

    Args:
        equity: Sermaye eğrisi.

    Returns:
        ``(max_dd_orani, dip_zamani, dd_suresi_bar)``.
    """
    if equity.empty:
        return 0.0, None, 0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    trough = dd.idxmin()
    mdd = float(dd.min())
    # Düşüşün toplam süresi: zirveden dibe.
    peak_time = equity.loc[:trough].idxmax() if trough is not None else None
    duration = int(equity.index.get_indexer([trough])[0] - equity.index.get_indexer([peak_time])[0]) if peak_time is not None else 0
    return mdd, trough, duration


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    bar_returns: pd.Series,
    cfg: BacktestConfig | None = None,
    data_cfg: DataConfig | None = None,
) -> dict[str, float]:
    """Backtest metriklerini hesaplar.

    **Sadece accuracy/isabet oranına bakmayın.** Yüksek isabet oranı, küçük
    kazançlar + büyük kayıplar anlamına gelebilir. Bu yüzden ortalama
    kazanç/kayıp, profit factor ve maliyet toplamı da raporlanır.

    Args:
        trades: İşlem tablosu.
        equity: Sermaye eğrisi.
        bar_returns: Bar bazında getiriler.
        cfg: Backtest konfigürasyonu.
        data_cfg: Yıllıklaştırma bilgisi.

    Returns:
        Metrik sözlüğü.
    """
    c = cfg or CONFIG.backtest
    dc = data_cfg or CONFIG.data
    bpy = dc.bars_per_year

    if trades.empty:
        return {"islem_sayisi": 0.0, "toplam_getiri": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "isabet_orani": float("nan")}

    net = trades["net_getiri"].to_numpy(dtype="float64")
    wins = net[net > 0]
    losses = net[net < 0]

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_bars = max(len(equity) - 1, 1)
    years = n_bars / bpy
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    # Sharpe/Sortino BAR frekansında değil, metrik frekansında (varsayılan
    # saatlik) hesaplanır — bkz. to_period_returns docstring'i.
    ppy = periods_per_year(c.metric_freq)
    r = to_period_returns(equity, c.metric_freq).to_numpy(dtype="float64")
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float((r.mean() - c.risk_free_rate) / sd * np.sqrt(ppy)) if sd > 0 else 0.0
    downside = r[r < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(r.mean() / dsd * np.sqrt(ppy)) if dsd > 0 else 0.0

    mdd, trough, dd_dur = max_drawdown(equity)
    gross_total = float(trades["brut_getiri"].mul(trades["size"]).sum())
    cost_total = float(trades["maliyet"].sum())

    return {
        "islem_sayisi": float(len(trades)),
        "toplam_getiri": total_return,
        "yillik_getiri_cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "dd_suresi_bar": float(dd_dur),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "isabet_orani": float((net > 0).mean()),
        "ort_kazanc": float(wins.mean()) if len(wins) else 0.0,
        "ort_kayip": float(losses.mean()) if len(losses) else 0.0,
        "kazanc_kayip_orani": float(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0
        else float("nan"),
        "profit_factor": float(wins.sum() / abs(losses.sum()))
        if len(losses) and losses.sum() != 0
        else float("nan"),
        "ort_islem_getirisi": float(net.mean()),
        "ort_tutma_bar": float(trades["tutma_bar"].mean()),
        "toplam_maliyet": cost_total,
        "maliyet_oncesi_getiri": gross_total,
        "maliyetin_brute_orani": float(cost_total / abs(gross_total))
        if gross_total != 0
        else float("nan"),
        "ort_pozisyon_buyuklugu": float(trades["size"].mean()),
        "piyasada_kalma_orani": float((trades["tutma_bar"].sum()) / max(len(equity), 1)),
    }


def buy_and_hold_benchmark(
    df: pd.DataFrame,
    cfg: BacktestConfig | None = None,
    data_cfg: DataConfig | None = None,
) -> dict[str, float]:
    """Al-ve-tut (buy & hold) karşılaştırma ölçütü.

    Stratejinin bundan iyi olmaması, tüm hattın boşa gittiği anlamına gelir.

    Args:
        df: OHLCV verisi.
        cfg: Backtest konfigürasyonu (giriş/çıkış maliyeti için).
        data_cfg: Yıllıklaştırma bilgisi.

    Returns:
        Ölçüt metrikleri.
    """
    c = cfg or CONFIG.backtest
    close = df["close"]
    equity = (close / close.iloc[0]) * c.initial_capital * (1.0 - total_cost_rate(c) / 2.0)
    ppy = periods_per_year(c.metric_freq)
    r = to_period_returns(equity, c.metric_freq).to_numpy()
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    mdd, _, _ = max_drawdown(equity)
    return {
        "toplam_getiri": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        "max_drawdown": mdd,
    }


def sensitivity_analysis(
    df: pd.DataFrame,
    events: pd.DataFrame,
    prob: pd.Series,
    cost_multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    cfg: BacktestConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Maliyet varsayımına duyarlılık analizi.

    Strateji ancak maliyetler 2 katına çıktığında da ayakta kalıyorsa
    güvenilirdir. ``0.0`` satırı, "maliyetsiz dünyada" performansı gösterir —
    aradaki uçurum, stratejinin maliyet duyarlılığını ele verir.

    Args:
        df: OHLCV verisi.
        events: Olay/etiket tablosu.
        prob: Örneklem-dışı olasılıklar.
        cost_multipliers: Baz maliyetin çarpanları.
        cfg: Baz backtest konfigürasyonu.
        **kwargs: :func:`run_backtest`'e geçirilecek ek argümanlar.

    Returns:
        Çarpan başına özet metrik tablosu.
    """
    import copy

    base = cfg or CONFIG.backtest
    rows = []
    for m in cost_multipliers:
        cc = copy.deepcopy(base)
        cc.commission_rate = base.commission_rate * m
        cc.slippage_bps = base.slippage_bps * m
        res = run_backtest(df, events, prob, cfg=cc, **kwargs)
        rows.append(
            {
                "maliyet_carpani": m,
                "islem_sayisi": res.metrics.get("islem_sayisi", 0.0),
                "toplam_getiri": res.metrics.get("toplam_getiri", 0.0),
                "sharpe": res.metrics.get("sharpe", 0.0),
                "max_drawdown": res.metrics.get("max_drawdown", 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_backtest(
    result: BacktestResult,
    path: str | Path,
    benchmark: pd.Series | None = None,
    title: str = "Maliyet sonrası backtest",
) -> Path:
    """Sermaye eğrisi, drawdown ve işlem getirisi dağılımını çizer.

    Args:
        result: Backtest sonucu.
        path: Hedef PNG yolu.
        benchmark: Karşılaştırma eğrisi (örn. al-ve-tut).
        title: Grafik başlığı.

    Returns:
        Yazılan dosyanın yolu.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[2, 1, 1])

    eq = result.equity
    axes[0].plot(eq.index, eq.to_numpy(), label="Strateji (maliyet sonrası)", color="#2b6cb0")
    if benchmark is not None:
        bm = benchmark.reindex(eq.index).ffill()
        axes[0].plot(bm.index, bm.to_numpy(), label="Al-ve-tut", color="#a0aec0", alpha=0.8)
    axes[0].set_title(title)
    axes[0].set_ylabel("Sermaye (TL)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    dd = eq / eq.cummax() - 1.0
    axes[1].fill_between(dd.index, dd.to_numpy(), 0.0, color="#c53030", alpha=0.5)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    if not result.trades.empty:
        axes[2].hist(
            result.trades["net_getiri"].to_numpy() * 100.0,
            bins=50,
            color="#2f855a",
            alpha=0.8,
        )
        axes[2].axvline(0.0, color="k", linewidth=1)
    axes[2].set_xlabel("İşlem net getirisi (%)")
    axes[2].set_ylabel("Adet")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p
