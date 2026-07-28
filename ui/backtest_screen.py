"""Backtest ekranı: walk-forward test, metrikler ve grafik."""
from __future__ import annotations

import streamlit as st

from fpredict import backtest, config
from . import common


def render():
    st.header("📊 Backtest (Geçmişe Dönük Test)")
    st.caption(
        "Her maç, YALNIZCA kendisinden önceki maçlarla eğitilmiş modelle tahmin "
        "edilir (veri sızıntısı yok). Böylece modelin gerçek öngörü gücü ölçülür."
    )

    bust = st.session_state.get("data_version", 0)
    from fpredict import db
    summary = db.league_summary()
    if summary.empty:
        st.info("Önce **Veri** sekmesinden lig indirin.")
        return

    name_map = {code: name for _, (name, code) in config.LEAGUES.items()}
    league_key = st.selectbox(
        "Lig", options=list(summary["league"]),
        format_func=lambda c: f"{name_map.get(c, c)} ({c})",
    )

    c1, c2, c3 = st.columns(3)
    half_life = c1.slider("Yarı-ömür (gün)", 30, 720, int(config.DEFAULT_HALF_LIFE_DAYS), 30)
    min_train = c2.number_input("Isınma (ilk N maç)", 100, 2000, 200, 50)
    refit_every = c3.number_input("Kaç maçta bir yeniden fit", 5, 100, 20, 5)

    st.caption(
        "Not: Backtest her yeniden-fit'te modeli baştan çözer; büyük veri + sık "
        "refit yavaş olabilir. Refit aralığını artırmak hızlandırır."
    )

    if not st.button("▶️ Backtest'i Çalıştır", type="primary"):
        return

    df = common.cached_matches(league_key, bust)
    prog = st.progress(0.0, text="Çalışıyor…")
    try:
        res = backtest.run_backtest(
            df, half_life=half_life, min_train=int(min_train),
            refit_every=int(refit_every), progress=lambda f: prog.progress(min(f, 1.0)),
        )
    except ValueError as exc:
        prog.empty()
        st.error(str(exc))
        return
    prog.empty()

    # --- Metrikler ---------------------------------------------------------- #
    st.subheader("Sonuçlar")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tahmin edilen maç", res.n_predicted)
    m2.metric("Doğruluk", f"%{res.accuracy*100:.1f}",
              delta=f"{(res.accuracy-res.baseline_home_acc)*100:+.1f} pp vs ev-sahibi")
    m3.metric("Log-loss", f"{res.log_loss:.3f}",
              delta=f"{res.log_loss-res.baseline_uniform_logloss:+.3f} vs 1/3",
              delta_color="inverse")
    m4.metric("Brier", f"{res.brier:.3f}")

    st.info(res.commentary())

    # --- Karşılaştırma grafiği --------------------------------------------- #
    _plot_comparison(res)

    # --- Kalibrasyon / kümülatif doğruluk ---------------------------------- #
    _plot_cumulative(res)

    with st.expander("Tahmin kayıtları (ilk 200)"):
        st.dataframe(res.records.head(200), use_container_width=True, hide_index=True)


def _plot_comparison(res):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    # log-loss karşılaştırması
    labels = ["Model", "Naif ev-sahibi", "Bilgisiz 1/3"]
    vals = [res.log_loss, res.baseline_home_logloss, res.baseline_uniform_logloss]
    axes[0].bar(labels, vals, color=["#2e7d32", "#9e9e9e", "#bdbdbd"])
    axes[0].set_title("Log-loss (düşük = iyi)")
    axes[0].tick_params(axis="x", rotation=15)

    accs = [res.accuracy, res.baseline_home_acc]
    axes[1].bar(["Model", "Hep ev-sahibi"], accs, color=["#1565c0", "#9e9e9e"])
    axes[1].set_title("Doğruluk (yüksek = iyi)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig)


def _plot_cumulative(res):
    import matplotlib.pyplot as plt
    import numpy as np

    if res.records.empty:
        return
    rec = res.records.copy()
    correct = (rec["pred"] == rec["actual"]).to_numpy(dtype=float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(rec["date"], cum_acc, color="#1565c0", label="Model kümülatif doğruluk")
    ax.axhline(res.baseline_home_acc, color="#9e9e9e", ls="--", label="Hep ev-sahibi")
    ax.set_ylabel("Kümülatif doğruluk")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
