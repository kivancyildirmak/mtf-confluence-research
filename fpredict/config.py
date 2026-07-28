"""Uygulama sabitleri: lig kodları, dosya yolları, varsayılan model parametreleri."""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Yollar
# --------------------------------------------------------------------------- #
# Cache ve ayarlar kullanıcının ev dizini altında saklanır; böylece PyInstaller
# ile paketlenmiş salt-okunur bir uygulama klasöründe bile yazma sorunu olmaz.
APP_DIR = Path(os.environ.get("FPREDICT_HOME", Path.home() / ".football_predictor"))
DB_PATH = APP_DIR / "cache.sqlite"
SETTINGS_PATH = APP_DIR / "settings.json"


def ensure_app_dir() -> Path:
    """Uygulama veri dizinini oluşturur (yoksa) ve döndürür."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    return APP_DIR


# --------------------------------------------------------------------------- #
# football-data.co.uk lig kodları
# --------------------------------------------------------------------------- #
# key -> (görünen ad, football-data.co.uk kodu)
LEAGUES: dict[str, tuple[str, str]] = {
    "T1": ("Türkiye Süper Lig", "T1"),
    "E0": ("İngiltere Premier Lig", "E0"),
    "E1": ("İngiltere Championship", "E1"),
    "SP1": ("İspanya La Liga", "SP1"),
    "D1": ("Almanya Bundesliga", "D1"),
    "I1": ("İtalya Serie A", "I1"),
    "F1": ("Fransa Ligue 1", "F1"),
    "N1": ("Hollanda Eredivisie", "N1"),
    "P1": ("Portekiz Primeira Liga", "P1"),
}

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data.org lig id eşlemesi (sakatlık/kadro API'si opsiyonel kullanım)
# API-Football (api-sports.io) league id'leri
APIFOOTBALL_LEAGUE_IDS: dict[str, int] = {
    "T1": 203,   # Süper Lig
    "E0": 39,    # Premier League
    "SP1": 140,  # La Liga
    "D1": 78,    # Bundesliga
    "I1": 135,   # Serie A
    "F1": 61,    # Ligue 1
    "N1": 88,    # Eredivisie
    "P1": 94,    # Primeira Liga
}

# --------------------------------------------------------------------------- #
# Model varsayılanları
# --------------------------------------------------------------------------- #
DEFAULT_HALF_LIFE_DAYS = 180.0   # zaman ağırlığı üstel azalma yarı-ömrü
DEFAULT_MAX_GOALS = 10           # skor matrisi boyutu (0..MAX_GOALS)
DEFAULT_SEASONS_BACK = 4         # kaç sezon otomatik indirilsin

# HTTP
HTTP_TIMEOUT = 30
USER_AGENT = "FootballPredictor/0.1 (+educational, statistical modelling)"
