@echo off
REM ===================================================================
REM  Futbol Tahmin - Windows baslatici
REM
REM  Bu dosyaya CIFT TIKLAYIN. Komut yazmaniza gerek yoktur.
REM  Masaustune kisayol icin: bu dosyaya sag tik -> Gonder -> Masaustu
REM ===================================================================

chcp 65001 >nul 2>nul
title Futbol Tahmin - Dixon-Coles

REM Bu .bat dosyasinin bulundugu klasore gec (ic ice klasor sorununu cozer)
cd /d "%~dp0"

REM --- app.py burada mi? ---------------------------------------------
if not exist "app.py" (
    echo.
    echo [HATA] app.py bu klasorde bulunamadi.
    echo Bu dosya, app.py ile AYNI klasorde olmalidir.
    echo Simdiki klasor: %CD%
    echo.
    pause
    exit /b 1
)

REM --- Python kurulu mu? ---------------------------------------------
python --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo [HATA] Python bulunamadi.
    echo.
    echo   1. https://www.python.org/downloads/ adresinden Python 3.11+ kurun
    echo   2. Kurulum ekraninda "Add python.exe to PATH" kutusunu ISARETLEYIN
    echo   3. Bu dosyayi tekrar calistirin
    echo.
    pause
    exit /b 1
)

REM --- Gerekli paketler kurulu mu? -----------------------------------
python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Gerekli paketler kuruluyor... Bu ilk seferde birkac dakika surer.
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [HATA] Paketler kurulamadi. Internet baglantinizi kontrol edin.
        pause
        exit /b 1
    )
)

REM --- Streamlit ilk calistirma e-posta sorusunu atla -----------------
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit" >nul 2>nul
    >"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

REM --- Baslat ---------------------------------------------------------
echo.
echo  Futbol Tahmin baslatiliyor...
echo  Tarayici birazdan otomatik acilacak (http://localhost:8501)
echo.
echo  KAPATMAK ICIN: bu pencereyi kapatin veya Ctrl+C
echo.

python -m streamlit run app.py --browser.gatherUsageStats=false

REM Hata ile kapandiysa pencere kapanmasin ki mesaj okunabilsin
if errorlevel 1 (
    echo.
    echo [HATA] Uygulama beklenmedik sekilde kapandi. Yukaridaki mesaji okuyun.
    pause
)
