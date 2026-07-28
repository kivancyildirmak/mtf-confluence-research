"""Futbol Maç Tahmin Uygulaması — Streamlit giriş noktası.

Dixon-Coles genişletmeli Poisson modeliyle 1/X/2, KG, Üst/Alt 2.5, en olası
skor ve beklenen gol tahminleri üretir. Veri football-data.co.uk'tan otomatik
indirilir ve yerel SQLite cache'te saklanır.

Çalıştırma:
    streamlit run app.py

YASAL / ETİK UYARI
------------------
Bu uygulama garanti sonuç vermez; yalnızca istatistiksel olasılık üretir. Bahis
risk içerir. Kaybetmeyi göze alamayacağınız parayı riske atmayın. Yalnızca
kullanım şartlarına uygun veri kaynakları kullanılmalıdır.
Bahis bağımlılığı desteği (Türkiye): ALO 191 (YEDAM / Yeşilay Danışmanlık).
"""
from __future__ import annotations

import streamlit as st

# Paketlerin import edilebilmesi için proje kökünü yola ekle (PyInstaller uyumu)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fpredict import config
from ui import (
    backtest_screen,
    data_screen,
    main_screen,
    squad_screen,
    value_screen,
)


def main():
    st.set_page_config(
        page_title="Futbol Tahmin — Dixon-Coles",
        page_icon="⚽",
        layout="wide",
    )
    config.ensure_app_dir()
    st.session_state.setdefault("data_version", 0)

    st.sidebar.title("⚽ Futbol Tahmin")
    st.sidebar.caption("Dixon-Coles / Poisson istatistik modeli")

    screen = st.sidebar.radio(
        "Ekran",
        ["Ana (Tahmin)", "Veri", "Kadro / Sakatlık", "Backtest", "Değer"],
    )

    st.sidebar.divider()
    st.sidebar.warning(
        "Bu araç garanti sonuç vermez, yalnızca istatistiksel olasılık üretir. "
        "Bahis risk içerir; kaybetmeyi göze alamayacağınızı riske atmayın.\n\n"
        "Bağımlılık desteği (TR): **ALO 191**."
    )

    if screen == "Ana (Tahmin)":
        main_screen.render()
    elif screen == "Veri":
        data_screen.render()
    elif screen == "Kadro / Sakatlık":
        squad_screen.render()
    elif screen == "Backtest":
        backtest_screen.render()
    elif screen == "Değer":
        value_screen.render()


if __name__ == "__main__":
    main()
