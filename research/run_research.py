"""Uçtan uca araştırma hattı.

Akış:

1. Veriyi yükle (yerel dosya veya sentetik).
2. Ortak modülle özellikleri üret (:mod:`features`).
3. Birincil kural + triple-barrier ile meta etiketleri üret (:mod:`labeling`).
4. Örnek ağırlıklarını hesapla (çakışan etiket düzeltmesi).
5. Purged K-Fold ile örneklem-dışı (OOS) olasılık üret.
6. Combinatorial Purged CV ile performans DAĞILIMI çıkar.
7. Walk-forward ile nihai doğrulama.
8. Maliyetli backtest + maliyet duyarlılık analizi.
9. Deflated Sharpe Ratio ile çoklu deneme düzeltmesi.
10. Nihai modeli tüm veriyle eğit, SHAP'ı hesapla, diske kaydet.

Çalıştırma:

    python -m research.run_research --synthetic
    python -m research.run_research --data data/btc_tl_1m.parquet

SIZINTI GÜVENCESİ: Backtest'e giren olasılıklar YALNIZCA örneklem-dışı
tahminlerdir (purged CV veya walk-forward). Örneklem-içi olasılıkla yapılan
backtest, mükemmele yakın sonuç verir ve tamamen anlamsızdır.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import (
    buy_and_hold_benchmark,
    periods_per_year,
    plot_backtest,
    run_backtest,
    sensitivity_analysis,
    total_cost_rate,
)
from .config import CONFIG, ResearchConfig, set_global_seed
from .data import build_synthetic_dataset, load_ohlcv
from .features import make_features, warmup_bars
from .labeling import (
    cusum_filter,
    get_meta_labels,
    get_sample_weights,
    get_target_volatility,
    label_summary,
    primary_side_rule,
)
from .model import (
    compute_shap_values,
    classification_metrics,
    feature_importance,
    plot_feature_importance,
    save_model,
    train_model,
)
from .validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe_from_returns,
    summarize_scores,
    walk_forward_splits,
)


def _log(msg: str) -> None:
    """Zaman damgalı ilerleme çıktısı."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1-4. Veri seti hazırlığı
# --------------------------------------------------------------------------- #


def prepare_dataset(
    cfg: ResearchConfig,
    data_path: str | None = None,
) -> dict[str, Any]:
    """Veriyi yükler, özellikleri ve etiketleri üretir, hizalar.

    Args:
        cfg: Kök konfigürasyon.
        data_path: Yerel veri dosyası. ``None`` ise sentetik veri üretilir.

    Returns:
        ``ohlcv``, ``X``, ``events``, ``weights``, ``target_vol`` anahtarlı sözlük.

    Raises:
        RuntimeError: Hizalama sonrası hiç olay kalmazsa.
    """
    if data_path:
        _log(f"Veri okunuyor: {data_path}")
        ohlcv = load_ohlcv(data_path)
        reference = orderbook = stable = None
    else:
        _log("Sentetik veri üretiliyor (Paribu API bağlı değil).")
        pack = build_synthetic_dataset(cfg)
        ohlcv = pack["ohlcv"]
        reference = pack["reference"]
        orderbook = pack["orderbook"]
        stable = pack["stable_premium"]
    _log(f"  {len(ohlcv):,} bar | {ohlcv.index[0]} -> {ohlcv.index[-1]}")

    _log("Özellikler üretiliyor (features.make_features)...")
    X = make_features(
        ohlcv,
        orderbook=orderbook,
        reference=reference,
        stable_premium=stable,
        cfg=cfg.features,
    )
    _log(f"  {X.shape[1]} özellik sütunu üretildi.")

    _log("Etiketler üretiliyor (triple-barrier + meta-label)...")
    target_vol = get_target_volatility(ohlcv["close"], cfg=cfg.labeling)
    side = primary_side_rule(ohlcv)
    event_idx = cusum_filter(
        ohlcv["close"], target_vol * cfg.labeling.cusum_threshold_mult
    )

    # Isınma dönemini at: özelliklerin çoğu henüz NaN. Bu filtre olmadan
    # model, güvenilmez özelliklerle eğitilir.
    warmup = warmup_bars(cfg.features)
    event_idx = event_idx[event_idx > ohlcv.index[min(warmup, len(ohlcv) - 1)]]

    events = get_meta_labels(
        ohlcv,
        primary_side=side,
        event_index=event_idx,
        target_vol=target_vol,
        cfg=cfg.labeling,
    )

    # Özellik/etiket hizalaması. Tamamen NaN olan satırlar atılır; kısmi NaN
    # LightGBM tarafından doğal olarak işlenir (mikroyapı sütunları gibi).
    Xe = X.reindex(events.index)
    usable = Xe.notna().mean(axis=1) > 0.5
    events = events.loc[usable.to_numpy()]
    Xe = Xe.loc[events.index]

    if events.empty:
        raise RuntimeError("Hizalama sonrası hiç olay kalmadı; parametreleri gözden geçirin.")

    _log(f"  {len(events):,} olay | pozitif oran {events['label'].mean():.3f}")

    _log("Örnek ağırlıkları hesaplanıyor (benzersizlik + getiri atfı + sönüm)...")
    weights = get_sample_weights(ohlcv["close"], events, cfg=cfg.labeling)
    _log(f"  ortalama benzersizlik: {weights['uniqueness'].mean():.3f}")

    return {
        "ohlcv": ohlcv,
        "X": Xe,
        "events": events,
        "weights": weights,
        "target_vol": target_vol,
        "label_summary": label_summary(events),
    }


# --------------------------------------------------------------------------- #
# 5. Purged K-Fold ile OOS olasılıklar
# --------------------------------------------------------------------------- #


def purged_cv_oos_probabilities(
    X: pd.DataFrame,  # noqa: N803
    events: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: ResearchConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Purged K-Fold ile tüm örneklem için OOS olasılık üretir.

    Args:
        X: Özellik matrisi (olay indeksli).
        events: Etiket tablosu.
        weights: Örnek ağırlıkları.
        cfg: Kök konfigürasyon.

    Returns:
        ``(oos_prob, fold_metrics)`` ikilisi.

    Notes:
        Her katmanda eğitim kümesi, test etiketleriyle çakışan örneklerden
        arındırılır (purging) ve test sonrasına embargo uygulanır. Erken
        durdurma KULLANILMAZ: doğrulama kümesi seçmek yeni bir sızıntı yüzeyi
        açar; ağaç sayısı config'ten sabittir.
    """
    y = events["label"]
    w = weights["weight"]
    cv = PurgedKFold(t1=events["t1"], n_splits=cfg.cv.n_splits, embargo_pct=cfg.cv.embargo_pct)

    oos = pd.Series(np.nan, index=X.index, name="prob")
    rows: list[dict[str, float]] = []

    for fold, (tr, te) in enumerate(cv.split(X), start=1):
        if len(tr) < 100 or len(np.unique(y.iloc[tr])) < 2:
            _log(f"  katman {fold}: yetersiz eğitim verisi, atlandı.")
            continue
        m = train_model(
            X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr],
            cfg=cfg.model, full_config=cfg,
        )
        p = m.predict_proba(X.iloc[te])
        oos.iloc[te] = p
        met = classification_metrics(y.iloc[te], p, threshold=cfg.backtest.prob_threshold)
        met["katman"] = float(fold)
        met["n_train"] = float(len(tr))
        met["purge_orani"] = 1.0 - len(tr) / max(len(X) - len(te), 1)
        rows.append(met)
        _log(
            f"  katman {fold}: n_test={len(te)} auc={met.get('auc', float('nan')):.4f} "
            f"acc={met['accuracy']:.4f} (purge ile atılan eğitim: {met['purge_orani']:.1%})"
        )

    return oos, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Combinatorial Purged CV
# --------------------------------------------------------------------------- #


def combinatorial_cv_distribution(
    ohlcv: pd.DataFrame,
    X: pd.DataFrame,  # noqa: N803
    events: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """Combinatorial Purged CV ile performans DAĞILIMI üretir.

    Her bölünmede model eğitilir, test grubunda OOS olasılık üretilir ve o
    olaylarla MALİYETLİ mini-backtest çalıştırılır. Böylece dağılım yalnızca
    AUC'de değil, gerçek para metriğinde (Sharpe, net getiri) de görülür.

    Args:
        ohlcv: Fiyat verisi.
        X: Özellik matrisi.
        events: Etiket tablosu.
        weights: Örnek ağırlıkları.
        cfg: Kök konfigürasyon.

    Returns:
        Bölünme başına metrik tablosu.
    """
    y = events["label"]
    w = weights["weight"]
    cpcv = CombinatorialPurgedCV(
        t1=events["t1"],
        n_groups=cfg.cv.cpcv_n_groups,
        n_test_groups=cfg.cv.cpcv_n_test_groups,
        embargo_pct=cfg.cv.embargo_pct,
    )
    _log(f"  {cpcv.n_splits} bölünme, {cpcv.n_paths} bağımsız backtest yolu.")

    rows: list[dict[str, float]] = []
    for i, (tr, te, combo) in enumerate(cpcv.split(X), start=1):
        if len(tr) < 100 or len(np.unique(y.iloc[tr])) < 2:
            continue
        m = train_model(
            X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr],
            cfg=cfg.model, full_config=cfg,
        )
        p = pd.Series(m.predict_proba(X.iloc[te]), index=X.index[te])
        met = classification_metrics(y.iloc[te], p, threshold=cfg.backtest.prob_threshold)

        res = run_backtest(
            ohlcv, events.loc[p.index], p,
            cfg=cfg.backtest, sizing_cfg=cfg.sizing,
            label_cfg=cfg.labeling, data_cfg=cfg.data,
        )
        rows.append(
            {
                "bolunme": float(i),
                "test_gruplari": str(combo),
                "auc": met.get("auc", float("nan")),
                "accuracy": met["accuracy"],
                "precision": met["precision"],
                "islem_sayisi": res.metrics.get("islem_sayisi", 0.0),
                "toplam_getiri": res.metrics.get("toplam_getiri", 0.0),
                "sharpe": res.metrics.get("sharpe", 0.0),
                "max_drawdown": res.metrics.get("max_drawdown", 0.0),
            }
        )
        if i % 5 == 0 or i == cpcv.n_splits:
            _log(f"  bölünme {i}/{cpcv.n_splits} tamamlandı.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7. Walk-forward
# --------------------------------------------------------------------------- #


def walk_forward_evaluation(
    X: pd.DataFrame,  # noqa: N803
    events: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: ResearchConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward (ileri yürüyüş) doğrulaması.

    Gerçek dağıtımın en yakın taklidi: geçmişle eğit, hemen sonraki dilimde
    test et, kaydır. CV'den farkı, hiçbir eğitim örneğinin testten SONRA
    gelmemesidir.

    Args:
        X: Özellik matrisi.
        events: Etiket tablosu.
        weights: Örnek ağırlıkları.
        cfg: Kök konfigürasyon.

    Returns:
        ``(wf_oos_prob, adim_metrikleri)``.
    """
    y = events["label"]
    w = weights["weight"]
    train_size = min(cfg.cv.wf_train_size, max(int(len(X) * 0.4), 100))
    test_size = min(cfg.cv.wf_test_size, max(int(len(X) * 0.1), 20))

    splits = walk_forward_splits(
        n_samples=len(X),
        train_size=train_size,
        test_size=test_size,
        step=min(cfg.cv.wf_step, test_size),
        expanding=cfg.cv.wf_expanding,
        t1=events["t1"],
        embargo_pct=cfg.cv.embargo_pct,
    )
    _log(f"  {len(splits)} walk-forward adımı (train={train_size}, test={test_size}).")

    oos = pd.Series(np.nan, index=X.index, name="prob")
    rows: list[dict[str, float]] = []
    for step, (tr, te) in enumerate(splits, start=1):
        if len(tr) < 100 or len(np.unique(y.iloc[tr])) < 2:
            continue
        m = train_model(
            X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr],
            cfg=cfg.model, full_config=cfg,
        )
        p = m.predict_proba(X.iloc[te])
        oos.iloc[te] = p
        met = classification_metrics(y.iloc[te], p, threshold=cfg.backtest.prob_threshold)
        met["adim"] = float(step)
        rows.append(met)
    return oos, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #


def run(
    cfg: ResearchConfig | None = None,
    data_path: str | None = None,
    output_dir: str | None = None,
    skip_shap: bool = False,
) -> dict[str, Any]:
    """Tüm araştırma hattını uçtan uca çalıştırır.

    Args:
        cfg: Kök konfigürasyon. ``None`` ise :data:`config.CONFIG`.
        data_path: Yerel veri dosyası. ``None`` ise sentetik veri.
        output_dir: Çıktı klasörü.
        skip_shap: SHAP hesabını atla (hız için).

    Returns:
        Tüm ara ve nihai sonuçları içeren sözlük.
    """
    c = cfg or CONFIG
    set_global_seed(c.seed)
    out_dir = Path(output_dir or c.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("PARIBU ML SİNYAL ARAŞTIRMA HATTI — offline (canlı API bağlı değil)")
    print("=" * 78)

    # --- 1-4 ---------------------------------------------------------------- #
    ds = prepare_dataset(c, data_path)
    ohlcv, X, events, weights = ds["ohlcv"], ds["X"], ds["events"], ds["weights"]

    print("\n--- Etiket özeti ---")
    print(ds["label_summary"].round(4).to_string(index=False))

    # --- 5 ------------------------------------------------------------------ #
    _log("Purged K-Fold CV (purging + embargo)...")
    oos_prob, fold_metrics = purged_cv_oos_probabilities(X, events, weights, c)
    covered = oos_prob.notna()
    _log(f"  OOS kapsama: {covered.mean():.1%} ({covered.sum():,} olay)")

    print("\n--- Purged K-Fold katman metrikleri ---")
    if not fold_metrics.empty:
        cols = ["katman", "n", "n_train", "auc", "accuracy", "precision", "recall", "brier"]
        print(fold_metrics.loc[:, [x for x in cols if x in fold_metrics]].round(4).to_string(index=False))

    # --- 6 ------------------------------------------------------------------ #
    _log("Combinatorial Purged CV (performans dağılımı)...")
    cpcv = combinatorial_cv_distribution(ohlcv, X, events, weights, c)
    print("\n--- Combinatorial Purged CV dağılımı ---")
    if not cpcv.empty:
        print(summarize_scores(cpcv.drop(columns=["test_gruplari"], errors="ignore")).to_string())

    # --- 7 ------------------------------------------------------------------ #
    _log("Walk-forward doğrulama...")
    wf_prob, wf_metrics = walk_forward_evaluation(X, events, weights, c)

    # --- 8 ------------------------------------------------------------------ #
    _log("Maliyetli backtest (komisyon + slippage + gecikme)...")
    cost_rt = total_cost_rate(c.backtest)
    _log(f"  gidiş-dönüş maliyet: {cost_rt * 100:.3f}% | gecikme: {c.backtest.latency_bars} bar")

    bt_cv = run_backtest(
        ohlcv, events.loc[covered], oos_prob.loc[covered],
        cfg=c.backtest, sizing_cfg=c.sizing, label_cfg=c.labeling, data_cfg=c.data,
    )
    wf_cov = wf_prob.notna()
    bt_wf = (
        run_backtest(
            ohlcv, events.loc[wf_cov], wf_prob.loc[wf_cov],
            cfg=c.backtest, sizing_cfg=c.sizing, label_cfg=c.labeling, data_cfg=c.data,
        )
        if wf_cov.any()
        else None
    )

    bh = buy_and_hold_benchmark(ohlcv, c.backtest, c.data)

    print("\n--- Backtest (Purged CV OOS olasılıklarıyla) ---")
    print(_metrics_table(bt_cv.metrics).to_string(index=False))
    if bt_wf is not None:
        print("\n--- Backtest (Walk-forward OOS olasılıklarıyla) ---")
        print(_metrics_table(bt_wf.metrics).to_string(index=False))
    print("\n--- Al-ve-tut ölçütü ---")
    print(_metrics_table(bh).to_string(index=False))

    _log("Maliyet duyarlılık analizi...")
    sens = sensitivity_analysis(
        ohlcv, events.loc[covered], oos_prob.loc[covered],
        cfg=c.backtest, sizing_cfg=c.sizing, label_cfg=c.labeling, data_cfg=c.data,
    )
    print("\n--- Maliyet duyarlılığı ---")
    print(sens.round(4).to_string(index=False))

    # --- 9 ------------------------------------------------------------------ #
    _log("Deflated Sharpe Ratio...")
    # Getiriler ve deneme Sharpe'ları AYNI frekansta olmalı (metrik frekansı).
    ppy = periods_per_year(c.backtest.metric_freq)
    trial_sharpes = cpcv["sharpe"].to_numpy() if not cpcv.empty else None
    dsr = deflated_sharpe_from_returns(
        bt_cv.period_returns,
        trial_sharpes=trial_sharpes,
        n_trials=c.cv.n_trials,
        periods_per_year=ppy,
        trial_sharpes_annualized=True,
        cfg=c.cv,
    )
    print("\n--- Deflated Sharpe Ratio ---")
    print(_metrics_table(dsr).to_string(index=False))
    print(
        "  Yorum: dsr < 0.95 ise, gözlenen Sharpe çoklu deneme sonrası "
        "istatistiksel olarak anlamlı SAYILMAZ."
    )

    # --- 10 ----------------------------------------------------------------- #
    _log("Nihai model tüm veriyle eğitiliyor...")
    final_model = train_model(
        X, events["label"], sample_weight=weights["weight"],
        cfg=c.model, full_config=c,
    )
    final_model.metadata.update(
        {
            "cv_ortalama_auc": float(fold_metrics["auc"].mean()) if not fold_metrics.empty else None,
            "cpcv_ortalama_sharpe": float(cpcv["sharpe"].mean()) if not cpcv.empty else None,
            "backtest_sharpe": bt_cv.metrics.get("sharpe"),
            "dsr": dsr.get("dsr"),
        }
    )

    shap_df = None
    imp = None
    if not skip_shap:
        _log("SHAP değerleri hesaplanıyor...")
        shap_df = compute_shap_values(
            final_model, X, max_samples=c.model.shap_max_samples, seed=c.seed
        )
        imp = feature_importance(final_model, shap_df)
        plot_feature_importance(imp, out_dir / "feature_importance.png", top_n=c.model.shap_top_n)
        print(f"\n--- En önemli {min(15, len(imp))} özellik (ortalama |SHAP|) ---")
        print(imp.head(15).round(6).to_string())
    else:
        imp = feature_importance(final_model)

    model_path = save_model(final_model, out_dir / "model.pkl")
    _log(f"Model kaydedildi: {model_path}")

    # --- Çıktılar ----------------------------------------------------------- #
    bh_curve = (ohlcv["close"] / ohlcv["close"].iloc[0]) * c.backtest.initial_capital
    plot_backtest(bt_cv, out_dir / "backtest.png", benchmark=bh_curve)
    if not cpcv.empty:
        _plot_distribution(cpcv, out_dir / "cpcv_distribution.png")

    _write_outputs(out_dir, c, ds, fold_metrics, cpcv, wf_metrics, bt_cv, sens, dsr, imp, bh)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"HAT TAMAMLANDI — {elapsed:.1f} sn | çıktılar: {out_dir.resolve()}")
    print("=" * 78)
    _print_verdict(bt_cv.metrics, dsr, cpcv, bh)

    return {
        "dataset": ds,
        "fold_metrics": fold_metrics,
        "cpcv": cpcv,
        "wf_metrics": wf_metrics,
        "backtest_cv": bt_cv,
        "backtest_wf": bt_wf,
        "sensitivity": sens,
        "dsr": dsr,
        "model": final_model,
        "importance": imp,
        "output_dir": out_dir,
    }


def _metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    """Metrik sözlüğünü okunabilir tabloya çevirir."""
    return pd.DataFrame(
        {"metrik": list(metrics.keys()), "deger": [_fmt(v) for v in metrics.values()]}
    )


def _fmt(v: Any) -> str:
    """Sayıları okunabilir biçimde yazar."""
    if isinstance(v, (int, float, np.floating)):
        if v != v:  # NaN
            return "nan"
        return f"{v:,.6f}" if abs(v) < 1000 else f"{v:,.2f}"
    return str(v)


def _plot_distribution(cpcv: pd.DataFrame, path: Path) -> Path:
    """CPCV Sharpe/getiri dağılımını çizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(
        axes, ["sharpe", "toplam_getiri"], ["Sharpe dağılımı", "Toplam getiri dağılımı"]
    ):
        vals = cpcv[col].to_numpy()
        ax.hist(vals, bins=max(5, len(vals) // 2), color="#2b6cb0", alpha=0.85)
        ax.axvline(0.0, color="k", linewidth=1)
        ax.axvline(float(np.mean(vals)), color="#c53030", linestyle="--", label="ortalama")
        ax.set_title(f"CPCV — {title}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_outputs(
    out_dir: Path,
    cfg: ResearchConfig,
    ds: dict[str, Any],
    fold_metrics: pd.DataFrame,
    cpcv: pd.DataFrame,
    wf_metrics: pd.DataFrame,
    bt: Any,
    sens: pd.DataFrame,
    dsr: dict[str, float],
    imp: pd.DataFrame | None,
    bh: dict[str, float],
) -> None:
    """Tüm tabloları ve deney kaydını diske yazar."""
    ds["label_summary"].to_csv(out_dir / "label_summary.csv", index=False)
    if not fold_metrics.empty:
        fold_metrics.to_csv(out_dir / "purged_cv_folds.csv", index=False)
    if not cpcv.empty:
        cpcv.to_csv(out_dir / "cpcv_distribution.csv", index=False)
        summarize_scores(cpcv.drop(columns=["test_gruplari"], errors="ignore")).to_csv(
            out_dir / "cpcv_summary.csv"
        )
    if not wf_metrics.empty:
        wf_metrics.to_csv(out_dir / "walk_forward.csv", index=False)
    if not bt.trades.empty:
        bt.trades.to_csv(out_dir / "trades.csv", index=False)
    sens.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    if imp is not None:
        imp.to_csv(out_dir / "feature_importance.csv")

    report = {
        "config": cfg.to_dict(),
        "backtest_metrics": bt.metrics,
        "backtest_varsayimlari": bt.config,
        "buy_and_hold": bh,
        "deflated_sharpe": dsr,
        "n_events": int(len(ds["events"])),
        "n_features": int(ds["X"].shape[1]),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _print_verdict(
    metrics: dict[str, float],
    dsr: dict[str, float],
    cpcv: pd.DataFrame,
    bh: dict[str, float],
) -> None:
    """Sonucu tek paragrafta, abartısız biçimde yorumlar."""
    print("\n--- KARAR ---")
    sharpe = metrics.get("sharpe", 0.0)
    n_trades = metrics.get("islem_sayisi", 0.0)
    checks = [
        (n_trades >= 30, f"işlem sayısı yeterli ({n_trades:.0f} >= 30)"),
        (sharpe > 0, f"maliyet sonrası Sharpe pozitif ({sharpe:.3f})"),
        (sharpe > bh.get("sharpe", 0.0), "al-ve-tut ölçütünü geçiyor"),
        (dsr.get("dsr", 0.0) > 0.95, f"DSR > 0.95 ({dsr.get('dsr', 0.0):.3f})"),
        (
            (not cpcv.empty) and float((cpcv["sharpe"] > 0).mean()) > 0.6,
            "CPCV yollarının >%60'ı pozitif",
        ),
    ]
    for ok, text in checks:
        print(f"  [{'GEÇTİ' if ok else 'KALDI'}] {text}")
    if all(ok for ok, _ in checks):
        print("  -> Tüm kontroller geçti. Yine de kağıt üzerinde (paper) test şarttır.")
    else:
        print(
            "  -> En az bir kontrol kaldı. Bu bir BAŞARISIZLIK DEĞİL, beklenen sonuçtur:\n"
            "     sentetik veride gerçek bir kenar (edge) yoktur ve hattın görevi zaten\n"
            "     olmayan kenarı 'var' göstermemektir."
        )


def main(argv: list[str] | None = None) -> int:
    """Komut satırı giriş noktası.

    Args:
        argv: Argüman listesi (test için).

    Returns:
        Çıkış kodu.
    """
    parser = argparse.ArgumentParser(
        description="Paribu ML sinyal araştırma hattı (offline, canlı API yok)."
    )
    parser.add_argument("--data", type=str, default=None, help="Yerel CSV/parquet yolu.")
    parser.add_argument("--synthetic", action="store_true", help="Sentetik veri kullan.")
    parser.add_argument("--bars", type=int, default=None, help="Sentetik bar sayısı.")
    parser.add_argument("--output", type=str, default=None, help="Çıktı klasörü.")
    parser.add_argument("--seed", type=int, default=None, help="Rastgelelik tohumu.")
    parser.add_argument("--skip-shap", action="store_true", help="SHAP hesabını atla.")
    args = parser.parse_args(argv)

    cfg = CONFIG
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.model.params["seed"] = args.seed
    if args.bars is not None:
        cfg.data.synthetic_bars = args.bars

    run(cfg=cfg, data_path=args.data, output_dir=args.output, skip_shap=args.skip_shap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
