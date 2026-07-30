"""Ana ekran: lig + takım seç, kadro ayarlarını uygula, tahmin göster."""
from __future__ import annotations

import streamlit as st

from fpredict import config, squad_adjust
from . import common


def _prob_bar(label: str, prob: float):
    st.markdown(f"**{label}** — %{prob*100:.1f}")
    st.progress(min(max(prob, 0.0), 1.0))


def render():
    st.header("⚽ Maç Tahmini")

    bust = st.session_state.get("data_version", 0)

    # --- Lig seçimi --------------------------------------------------------- #
    from fpredict import db
    summary = db.league_summary()
    available = list(summary["league"]) if not summary.empty else []
    if not available:
        st.info("Önce **Veri** sekmesinden en az bir lig indirin.")
        return

    name_map = config.LEAGUE_NAMES
    league_key = st.selectbox(
        "Lig", options=available,
        format_func=lambda c: f"{name_map.get(c, c)} ({c})",
    )

    teams = common.teams_for_league(league_key, bust)
    if len(teams) < 2:
        st.warning("Bu ligde yeterli takım yok. Veriyi güncelleyin.")
        return

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Ev Sahibi", teams, index=0)
    with col2:
        away_opts = [t for t in teams if t != home]
        away = st.selectbox("✈️ Deplasman", away_opts, index=0)

    half_life = st.slider(
        "Zaman ağırlığı yarı-ömrü (gün)", min_value=30, max_value=720,
        value=int(config.DEFAULT_HALF_LIFE_DAYS), step=30,
        help="Küçük değer = son form daha baskın. Büyük değer = uzun dönem güç.",
    )

    # --- Kadro ayarları (Kadro ekranında girilen) --------------------------- #
    adjustments = st.session_state.get("squad_adjustments", {})
    home_adj = adjustments.get(home, squad_adjust.SquadAdjustment())
    away_adj = adjustments.get(away, squad_adjust.SquadAdjustment())
    boosts = squad_adjust.apply_to_boosts(home_adj, away_adj)
    if boosts["home_boost"] != 1.0 or boosts["away_boost"] != 1.0:
        st.caption(
            f"ℹ️ Kadro güç çarpanları uygulandı — ev: ×{boosts['home_boost']:.2f}, "
            f"deplasman: ×{boosts['away_boost']:.2f} (kaba yaklaşım)."
        )

    if not st.button("🎯 Tahmin Et", type="primary"):
        return

    # --- Fit + tahmin ------------------------------------------------------- #
    df = common.cached_matches(league_key, bust)
    try:
        model_obj = common.cached_model(league_key, half_life, len(df), bust)
    except Exception as exc:
        st.error(f"Model fit edilemedi: {exc}")
        return

    try:
        pred = model_obj.predict(home, away, **boosts)
    except KeyError as exc:
        st.error(f"Tahmin yapılamadı: {exc}")
        return

    _render_prediction(pred)


def _render_prediction(pred: dict):
    st.divider()
    st.subheader(f"{pred['home']}  vs  {pred['away']}")

    # Beklenen goller + en olası skor
    c1, c2, c3 = st.columns(3)
    c1.metric("Beklenen gol (ev)", f"{pred['exp_home_goals']:.2f}")
    c2.metric("Beklenen gol (dep)", f"{pred['exp_away_goals']:.2f}")
    mls = pred["most_likely_score"]
    c3.metric("En olası skor", f"{mls[0]}–{mls[1]}")

    st.markdown("### Maç Sonucu (1 / X / 2)")
    _prob_bar(f"1 — {pred['home']}", pred["prob_home"])
    _prob_bar("X — Beraberlik", pred["prob_draw"])
    _prob_bar(f"2 — {pred['away']}", pred["prob_away"])

    cA, cB = st.columns(2)
    with cA:
        st.markdown("### Karşılıklı Gol (KG)")
        _prob_bar("KG Var", pred["prob_btts_yes"])
        _prob_bar("KG Yok", pred["prob_btts_no"])
    with cB:
        st.markdown("### Toplam Gol 2.5")
        _prob_bar("Üst 2.5", pred["prob_over25"])
        _prob_bar("Alt 2.5", pred["prob_under25"])

    with st.expander("🔢 Skor olasılık matrisi (ev satır, deplasman sütun)"):
        import pandas as pd
        mat = pred["score_matrix"]
        # ilk 6x6 yeterli
        k = min(6, mat.shape[0])
        show = pd.DataFrame(
            (mat[:k, :k] * 100).round(1),
            index=[f"ev {i}" for i in range(k)],
            columns=[f"dep {j}" for j in range(k)],
        )
        st.dataframe(show, use_container_width=True)
        st.caption("Değerler yüzde (%). En koyu hücre en olası skordur.")

    st.warning(
        "⚠️ Bu bir **istatistiksel olasılık tahminidir**, garanti sonuç değildir. "
        "Bahis risk içerir; kaybetmeyi göze alamayacağınız parayı riske atmayın."
    )
