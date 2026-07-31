@echo off
REM ===================================================================
REM  Masaustune "Futbol Tahmin" kisayolu olusturur.
REM  Bir kez calistirmaniz yeterlidir; sonrasinda masaustundeki
REM  kisayola cift tiklayarak uygulamayi acabilirsiniz.
REM ===================================================================

chcp 65001 >nul 2>nul
cd /d "%~dp0"

if not exist "Baslat.bat" (
    echo [HATA] Baslat.bat bulunamadi. Bu dosyayi proje klasorunde calistirin.
    pause
    exit /b 1
)

set "TARGET=%CD%\Baslat.bat"
set "WORKDIR=%CD%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'Futbol Tahmin.lnk')); $s.TargetPath=$env:TARGET; $s.WorkingDirectory=$env:WORKDIR; $s.Description='Futbol Tahmin - Dixon-Coles'; $s.Save()"

if errorlevel 1 (
    echo.
    echo [HATA] Kisayol olusturulamadi.
    echo Alternatif yontem: Baslat.bat dosyasina sag tik -^> Gonder -^> Masaustu
    pause
    exit /b 1
)

echo.
echo  Masaustune "Futbol Tahmin" kisayolu olusturuldu.
echo  Artik oradan cift tiklayarak acabilirsiniz.
echo.
pause
