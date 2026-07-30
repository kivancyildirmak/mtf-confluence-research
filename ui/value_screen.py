"""Değer ekranı: model olasılıkları vs bahis oranlarının ima olasılıkları."""
from __future__ import annotations

import streamlit as st

from fpredict import config, value
from . import common


def render():
    st.header("💎 Değer (Value) Analizi")
    st.error(
        "⚠️ **RİSK UYARISI:** 'Değerli' etiketi kazanç garantisi DEĞİLDİR. "
        "Model yanlış olabilir; oranlar piyasa tarafından verimli fiyatlanmış "
        "olabilir. Bu ekran yalnızca eğitim/analiz amaçlıdır. Bahis bağımlılığı "
        "için destek: **ALO 191** (Türkiye Uyuşturucu ve Bağımlılıkla Mücadele)."
    )
    st.caption(
        "Model olasılıkları, CSV'deki (Bet365 vb.) oranların **marj arındırılmış** "
        "ima olasılıklarıyla kıyaslanır. Model, ima olasılıktan belirgin yüksek "
        "olasılık verdiği bahisleri 'değerli' olarak işaretler (pozitif beklenen değer)."
    )

    bust = st.session_state.get("data_version", 0)
    from fpredict import db
    summary = db.league_summary()
    if summary.empty:
        st.info("Önce **Veri** sekmesinden lig indirin.")
        return

    name_map = config.LEAGUE_NAMES
    league_key = st.selectbox(
        "Lig", options=list(summary["league"]),
        format_func=lambda c: f"{name_map.get(c, c)} ({c})",
    )

    c1, c2 = st.columns(2)
    half_life = c1.slider("Yarı-ömür (gün)", 30, 720, int(config.DEFAULT_HALF_LIFE_DAYS), 30)
    min_edge = c2.slider("Minimum edge (model − ima)", 0.02, 0.20, 0.05, 0.01)

    st.markdown(
        "Bu analiz, cache'deki **son oynanmış** maçların oranlarını kullanır "
        "(gelecek maç oranları football-data CSV'sinde yer almaz). Yani modelin "
        "geçmiş oranlara göre nerede farklı düşündüğünü gösterir — bir tür "
        "tutarlılık/kalibrasyon incelemesi."
    )
    n_recent = st.number_input("Son kaç maç incelensin", 20, 500, 100, 20)

    if not st.button("🔍 Değerli Bahisleri Bul", type="primary"):
        return

    df = common.cached_matches(league_key, bust)
    if df.empty:
        st.warning("Veri yok.")
        return

    try:
        model_obj = common.cached_model(league_key, half_life, len(df), bust)
    except Exception as exc:
        st.error(f"Model fit edilemedi: {exc}")
        return

    recent = df.sort_values("date").tail(int(n_recent))
    preds = []
    skipped = 0
    for _, r in recent.iterrows():
        if r["home_team"] not in model_obj.attack or r["away_team"] not in model_obj.attack:
            skipped += 1
            continue
        if not r.get("odds_h") or not r.get("odds_d") or not r.get("odds_a"):
            skipped += 1
            continue
        p = model_obj.predict(r["home_team"], r["away_team"])
        p["odds_h"], p["odds_d"], p["odds_a"] = r["odds_h"], r["odds_d"], r["odds_a"]
        preds.append(p)

    vb = value.find_value_bets(preds, min_edge=min_edge)
    st.caption(f"{len(preds)} maç değerlendirildi, {skipped} maç atlandı (oran/takım eksik).")

    if vb.empty:
        st.success("Seçilen eşikte 'değerli' bahis bulunamadı. (Piyasa verimli görünüyor.)")
    else:
        st.subheader(f"{len(vb)} değerli bahis bulundu")
        st.dataframe(vb, use_container_width=True, hide_index=True)
        st.caption(
            "EV_% = 1 birim bahis başına beklenen yüzde getiri (modelin olasılığı "
            "doğruysa). Yüksek EV her zaman yüksek güven demek değildir."
        )
