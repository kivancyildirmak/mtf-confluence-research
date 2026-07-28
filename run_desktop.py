"""Masaüstü başlatıcı — PyInstaller ile paketlenen tek dosyalık uygulamanın
giriş noktası.

Streamlit normalde `streamlit run app.py` ile çalışır. Paketlenmiş bir exe'de
komut satırı olmadığından, Streamlit'i programatik olarak başlatır ve
tarayıcıyı otomatik açarız.

PyInstaller ipuçları için README.md > "Paketleme" bölümüne bakın.
"""
from __future__ import annotations

import os
import sys


def _resource_path(rel: str) -> str:
    """PyInstaller onefile modunda dosyalar geçici _MEIPASS dizinine açılır."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def main():
    # Streamlit'in kendi CLI'ını programatik çağır
    import streamlit.web.cli as stcli

    app_path = _resource_path("app.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=false",   # tarayıcıyı otomatik aç
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
