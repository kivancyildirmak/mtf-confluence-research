"""Veri ekranı: cache durumu, lig güncelleme, elle CSV yükleme (yedek)."""
from __future__ import annotations

import streamlit as st

from fpredict import config, data_fetch, db
from . import common


def render():
    st.header("📥 Veri Yönetimi")
    st.caption(
        "Veriler [football-data.co.uk](https://www.football-data.co.uk/) üzerinden "
        "otomatik indirilir ve yerel SQLite cache'e yazılır. Her açılışta baştan "
        "indirilmez; yalnızca güncelleme butonuna basınca yeni maçlar eklenir."
    )

    # --- Cache özeti --------------------------------------------------------- #
    st.subheader("Cache Durumu")
    summary = db.league_summary()
    if summary.empty:
        st.info("Cache boş. Aşağıdan bir lig seçip **Verileri Güncelle**'ye basın.")
    else:
        # okunur lig adları ekle
        name_map = {code: name for _, (name, code) in config.LEAGUES.items()}
        summary.insert(1, "lig_adı", summary["league"].map(name_map).fillna(summary["league"]))
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # --- Güncelleme --------------------------------------------------------- #
    st.subheader("Verileri Güncelle")
    col1, col2 = st.columns([2, 1])
    with col1:
        league_key = st.selectbox(
            "Lig",
            options=list(config.LEAGUES.keys()),
            format_func=lambda k: f"{config.LEAGUES[k][0]} ({k})",
        )
    with col2:
        seasons_back = st.number_input(
            "Kaç sezon", min_value=1, max_value=15,
            value=config.DEFAULT_SEASONS_BACK,
        )

    if st.button("🔄 Verileri Güncelle", type="primary"):
        prog = st.progress(0.0, text="Başlatılıyor…")

        def _cb(msg, frac):
            prog.progress(min(frac, 1.0), text=msg)

        try:
            result = data_fetch.update_league(
                league_key, seasons_back=int(seasons_back), progress=_cb
            )
            common.bump_data_version()
            prog.empty()
            if result["seasons_ok"]:
                st.success(
                    f"Güncellendi: {result['inserted']} maç işlendi. "
                    f"Başarılı sezonlar: {', '.join(result['seasons_ok'])}."
                )
            if result["seasons_failed"]:
                st.warning(
                    "Bazı sezonlar indirilemedi: "
                    f"{', '.join(result['seasons_failed'])}.\n\n"
                    + "\n".join(f"• {e}" for e in result["errors"])
                )
                _manual_upload_fallback(league_key)
        except data_fetch.DataFetchError as exc:
            prog.empty()
            st.error(f"İndirme hatası: {exc}")
            _manual_upload_fallback(league_key)
        except Exception as exc:  # beklenmeyen
            prog.empty()
            st.error(f"Beklenmeyen hata: {exc}")

    st.divider()
    with st.expander("📄 Elle CSV Yükle (yedek yöntem)"):
        _manual_upload_fallback(league_key, standalone=True)


def _manual_upload_fallback(league_key: str, standalone: bool = False):
    """İndirme başarısızsa kullanıcı football-data CSV'sini elle yükleyebilir."""
    if not standalone:
        st.markdown("**İnternet erişimi yoksa:** CSV'yi elle yükleyebilirsiniz.")
    _, code = config.LEAGUES[league_key]
    up = st.file_uploader(
        f"{config.LEAGUES[league_key][0]} için football-data.co.uk CSV'si",
        type=["csv"], key=f"upload_{league_key}_{standalone}",
    )
    if up is not None:
        try:
            rows = data_fetch.parse_csv_bytes(up.getvalue(), code)
            n = db.upsert_matches(rows)
            db.touch_updated(code)
            common.bump_data_version()
            st.success(f"{n} maç yüklendi ve cache'e eklendi.")
        except data_fetch.DataFetchError as exc:
            st.error(f"Dosya işlenemedi: {exc}")
