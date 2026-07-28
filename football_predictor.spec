# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — tek dosyalık masaüstü paketi üretir.

Kullanım:
    pip install pyinstaller
    pyinstaller football_predictor.spec

Streamlit metadata ve statik dosyaları toplamak için collect_all kullanılır;
aksi halde paketlenen exe "streamlit.web" veya metadata bulunamadı hataları
verir.
"""
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

# Streamlit ve alt paketlerini eksiksiz topla
for pkg in ("streamlit", "altair", "pyarrow"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Sürüm metadata'sı (Streamlit çalışma zamanında okur)
for pkg in ("streamlit", "pandas", "numpy", "scipy", "matplotlib"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Uygulama kaynak dosyaları
datas += [
    ("app.py", "."),
    ("fpredict", "fpredict"),
    ("ui", "ui"),
    ("sample_data", "sample_data"),
]

hiddenimports += ["fpredict", "ui", "scipy.special", "scipy.optimize"]


block_cipher = None

a = Analysis(
    ["run_desktop.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="FootballPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,   # Streamlit çıktısı/portu görebilmek için konsol açık
)
