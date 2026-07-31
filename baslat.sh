#!/usr/bin/env bash
# Futbol Tahmin — macOS / Linux başlatıcı
#
# Kullanım:  ./baslat.sh      (ilk seferde: chmod +x baslat.sh)
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f app.py ]; then
    echo "[HATA] app.py bu klasörde bulunamadı: $(pwd)"
    exit 1
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[HATA] python3 bulunamadı. Python 3.11+ kurun."
    exit 1
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
    echo "Gerekli paketler kuruluyor (ilk seferde birkaç dakika sürer)…"
    "$PY" -m pip install -r requirements.txt
fi

# Streamlit ilk çalıştırma e-posta sorusunu atla
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo
echo " Futbol Tahmin başlatılıyor — tarayıcı otomatik açılacak."
echo " Kapatmak için: Ctrl+C"
echo
exec "$PY" -m streamlit run app.py --browser.gatherUsageStats=false
