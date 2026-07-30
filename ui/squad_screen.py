"""Kadro/Sakatlık ekranı: API'den çekme veya elle güç çarpanı girme."""
from __future__ import annotations

import streamlit as st

from fpredict import config, squad_adjust
from . import common


def render():
    st.header("🩺 Kadro / Sakatlık Ayarları")
    st.caption(
        "Sakat/cezalı oyuncuların etkisini modele **güç çarpanı** olarak yansıtın. "
        "Bu düzeltme kaba bir yaklaşımdır: gerçek oyuncu katkısını (dakika, xG, "
        "pozisyon) hesaba katmaz, yalnızca beklenen golü ölçekler."
    )

    st.session_state.setdefault("squad_adjustments", {})

    # --- API ayarları ------------------------------------------------------- #
    with st.expander("🔑 API-Football anahtarı (opsiyonel, otomatik sakatlık çekimi)"):
        st.markdown(
            "[api-sports.io](https://www.api-sports.io/) ücretsiz kotalı bir anahtar "
            "verir. Anahtar yerelde `settings.json` içinde saklanır, hiçbir yere "
            "gönderilmez."
        )
        current = squad_adjust.load_api_key() or ""
        key_in = st.text_input("API anahtarı", value=current, type="password")
        if st.button("Anahtarı Kaydet"):
            squad_adjust.save_api_key(key_in.strip())
            st.success("Anahtar kaydedildi.")

    st.divider()

    # --- Lig / takım seçimi ------------------------------------------------- #
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
    teams = common.teams_for_league(league_key, bust)
    if not teams:
        return
    team = st.selectbox("Takım", teams)

    adj = st.session_state["squad_adjustments"].get(team, squad_adjust.SquadAdjustment())

    # --- API'den sakatlık çek ---------------------------------------------- #
    api_key = squad_adjust.load_api_key()
    if api_key:
        from fpredict.api_football import api_seasons

        season = api_seasons(config.key_for_code(league_key), 1)[0]
        if st.button(f"📡 '{team}' için sakatlıkları çek ({season} sezonu)"):
            with st.spinner("API sorgulanıyor…"):
                try:
                    injuries = squad_adjust.fetch_injuries_for_team(
                        api_key, config.key_for_code(league_key), team, season
                    )
                except squad_adjust.SquadAPIError as exc:
                    st.error(f"Sakatlık verisi alınamadı: {exc}")
                    injuries = None

            if injuries is not None:
                if not injuries:
                    st.info(
                        f"API, '{team}' için bu sezon kayıtlı sakat/cezalı oyuncu "
                        "döndürmedi. (Kayıt olmaması sakat oyuncu yok demek "
                        "olmayabilir.)"
                    )
                else:
                    st.success(f"{len(injuries)} kayıt bulundu:")
                    import pandas as pd
                    st.dataframe(pd.DataFrame(injuries), use_container_width=True,
                                 hide_index=True)
                    st.caption(
                        "Bu liste bilgilendirme amaçlıdır. Etkiyi modele yansıtmak "
                        "için aşağıdan güç çarpanını ayarlayın — hangi oyuncunun ne "
                        "kadar kritik olduğunu uygulama bilemez."
                    )
                    st.session_state[f"inj_count_{team}"] = len(injuries)
    else:
        st.caption(
            "ℹ️ Otomatik sakatlık çekimi için **Veri** ekranından API-Football "
            "anahtarı girin. Anahtarsız da aşağıdan elle çarpan girebilirsiniz."
        )

    # --- Elle giriş --------------------------------------------------------- #
    st.subheader(f"'{team}' için güç çarpanı")
    mode = st.radio(
        "Yöntem", ["Doğrudan çarpan", "Eksik oyuncu sayısından hesapla"],
        horizontal=True,
    )

    if mode == "Doğrudan çarpan":
        atk = st.slider("Hücum çarpanı", 0.5, 1.2, float(adj.attack_mult), 0.01,
                        help="1.0 = etkisiz. <1 daha az gol atar.")
        dfn = st.slider("Savunma zayıflığı çarpanı", 0.8, 1.5, float(adj.defense_mult), 0.01,
                        help="1.0 = etkisiz. >1 rakip daha çok gol atar.")
    else:
        n_missing = st.number_input("Eksik kilit oyuncu sayısı", 0, 11, 0)
        penalty = st.slider("Oyuncu başına ceza", 0.02, 0.15, 0.06, 0.01)
        atk = squad_adjust.multiplier_from_missing(int(n_missing), penalty)
        dfn = 1.0 + (1.0 - atk) * 0.5  # savunmaya yarı etki
        st.caption(f"Hesaplanan → hücum ×{atk:.2f}, savunma zayıflığı ×{dfn:.2f}")

    note = st.text_input("Not (opsiyonel)", value=adj.note)

    if st.button("💾 Bu takım için kaydet", type="primary"):
        st.session_state["squad_adjustments"][team] = squad_adjust.SquadAdjustment(
            attack_mult=atk, defense_mult=dfn, note=note,
        )
        st.success(f"'{team}' ayarı kaydedildi. Ana ekrandaki tahminlere yansıyacak.")

    # --- Mevcut ayarlar ----------------------------------------------------- #
    adjustments = st.session_state["squad_adjustments"]
    if adjustments:
        st.divider()
        st.subheader("Kayıtlı ayarlar")
        for t, a in adjustments.items():
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(f"**{t}** {('— ' + a.note) if a.note else ''}")
            cols[1].write(f"hücum ×{a.attack_mult:.2f}")
            cols[2].write(f"savunma ×{a.defense_mult:.2f}")
            if cols[3].button("Sil", key=f"del_{t}"):
                del st.session_state["squad_adjustments"][t]
                st.rerun()
