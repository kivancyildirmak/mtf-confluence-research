"""Uygulama sabitleri: lig kodları, dosya yolları, varsayılan model parametreleri."""
from __future__ import annotations

import os
from dataclasses import dataclass
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
# football-data.co.uk verileri İKİ farklı biçimde yayınlar:
#
#   "main"  -> https://www.football-data.co.uk/mmz4281/{SEZON}/{KOD}.csv
#              Her sezon ayrı dosya. Sütunlar: Date, HomeTeam, AwayTeam,
#              FTHG, FTAG, FTR, B365H/D/A ...
#              (Büyük Avrupa ligleri: İngiltere, Almanya, İspanya, İtalya,
#               Fransa, Hollanda, Belçika, Portekiz, Türkiye, Yunanistan, İskoçya)
#
#   "extra" -> https://www.football-data.co.uk/new/{KOD}.csv
#              TÜM sezonlar tek dosyada. Sütunlar farklı: Date, Home, Away,
#              HG, AG, Res, PH/PD/PA, AvgH/AvgD/AvgA ...
#              (İskandinavya, Polonya, Avusturya, İrlanda, ABD, Brezilya vb.)
#
# Bu yüzden her lig için hangi kaynağın kullanılacağını `source` alanı belirler.


@dataclass(frozen=True)
class LeagueInfo:
    """Bir ligin görünen adı, football-data kodu ve veri kaynağı biçimi."""
    name: str          # arayüzde görünen ad
    code: str          # football-data.co.uk dosya kodu
    source: str        # "main" (sezon başına dosya) veya "extra" (tek dosya)
    country: str = ""  # gruplama için


LEAGUES: dict[str, LeagueInfo] = {
    # ---- "main" biçim: mmz4281/{sezon}/{kod}.csv -------------------------- #
    "T1":  LeagueInfo("Türkiye Süper Lig", "T1", "main", "Türkiye"),
    "E0":  LeagueInfo("İngiltere Premier Lig", "E0", "main", "İngiltere"),
    "E1":  LeagueInfo("İngiltere Championship", "E1", "main", "İngiltere"),
    "SC0": LeagueInfo("İskoçya Premiership", "SC0", "main", "İskoçya"),
    "SP1": LeagueInfo("İspanya La Liga", "SP1", "main", "İspanya"),
    "SP2": LeagueInfo("İspanya La Liga 2", "SP2", "main", "İspanya"),
    "D1":  LeagueInfo("Almanya Bundesliga", "D1", "main", "Almanya"),
    "D2":  LeagueInfo("Almanya 2. Bundesliga", "D2", "main", "Almanya"),
    "I1":  LeagueInfo("İtalya Serie A", "I1", "main", "İtalya"),
    "I2":  LeagueInfo("İtalya Serie B", "I2", "main", "İtalya"),
    "F1":  LeagueInfo("Fransa Ligue 1", "F1", "main", "Fransa"),
    "F2":  LeagueInfo("Fransa Ligue 2", "F2", "main", "Fransa"),
    "N1":  LeagueInfo("Hollanda Eredivisie", "N1", "main", "Hollanda"),
    "B1":  LeagueInfo("Belçika Jupiler Pro Lig", "B1", "main", "Belçika"),
    "P1":  LeagueInfo("Portekiz Primeira Liga", "P1", "main", "Portekiz"),
    "G1":  LeagueInfo("Yunanistan Super Lig", "G1", "main", "Yunanistan"),

    # ---- "extra" biçim: new/{kod}.csv ------------------------------------- #
    # Not: İskandinav ligleri ilkbahar-sonbahar takvimiyle oynanır.
    "SWE": LeagueInfo("İsveç Allsvenskan", "SWE", "extra", "İsveç"),
    "NOR": LeagueInfo("Norveç Eliteserien", "NOR", "extra", "Norveç"),
    "DNK": LeagueInfo("Danimarka Superliga", "DNK", "extra", "Danimarka"),
    "FIN": LeagueInfo("Finlandiya Veikkausliiga", "FIN", "extra", "Finlandiya"),
    "POL": LeagueInfo("Polonya Ekstraklasa", "POL", "extra", "Polonya"),
    "AUT": LeagueInfo("Avusturya Bundesliga", "AUT", "extra", "Avusturya"),
    "SWZ": LeagueInfo("İsviçre Super Lig", "SWZ", "extra", "İsviçre"),
    "IRL": LeagueInfo("İrlanda Premier Division", "IRL", "extra", "İrlanda"),
    "ROU": LeagueInfo("Romanya Liga 1", "ROU", "extra", "Romanya"),
    "RUS": LeagueInfo("Rusya Premier Lig", "RUS", "extra", "Rusya"),
    "USA": LeagueInfo("ABD MLS", "USA", "extra", "ABD"),
    "MEX": LeagueInfo("Meksika Liga MX", "MEX", "extra", "Meksika"),
    "BRA": LeagueInfo("Brezilya Serie A", "BRA", "extra", "Brezilya"),
    "ARG": LeagueInfo("Arjantin Primera Division", "ARG", "extra", "Arjantin"),
    "JPN": LeagueInfo("Japonya J1 Lig", "JPN", "extra", "Japonya"),
    "CHN": LeagueInfo("Çin Super Lig", "CHN", "extra", "Çin"),
}

# kod -> görünen ad (arayüzde cache tablosunu okunur hale getirmek için)
LEAGUE_NAMES: dict[str, str] = {info.code: info.name for info in LEAGUES.values()}


def league_label(key: str) -> str:
    """Arayüz seçim kutusu etiketi."""
    info = LEAGUES.get(key)
    return f"{info.name} ({key})" if info else key


def league_name_for_code(code: str) -> str:
    """football-data kodundan görünen adı döndürür (bilinmiyorsa kodun kendisi)."""
    return LEAGUE_NAMES.get(code, code)


BASE_URL = "https://www.football-data.co.uk/mmz4281"
EXTRA_BASE_URL = "https://www.football-data.co.uk/new"

# --------------------------------------------------------------------------- #
# Yedek (ayna) kaynak
# --------------------------------------------------------------------------- #
# Bazı ağlarda (ISS engeli, kurumsal filtre, antivirüs HTTPS taraması)
# football-data.co.uk'a erişilemez. Bu durumda aşağıdaki GitHub aynası denenir.
# Ayna, football-data.co.uk ile AYNI sütun düzenini kullanır (HomeTeam/FTHG/...),
# ancak yalnızca 5 büyük ligi kapsar ve BAHİS ORANI SÜTUNLARI YOKTUR.
MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/football-datasets/main"
    "/datasets/{dir}/season-{season}.csv"
)

# lig kodu -> aynadaki klasör adı (yalnızca aynanın kapsadığı ligler)
MIRROR_DIRS: dict[str, str] = {
    "E0": "premier-league",
    "SP1": "la-liga",
    "I1": "serie-a",
    "D1": "bundesliga",
    "F1": "ligue-1",
}


def has_mirror(league_code: str) -> bool:
    """Bu lig için yedek ayna kaynağı var mı?"""
    return league_code in MIRROR_DIRS


def mirror_url(league_code: str, season: str) -> str | None:
    """Ayna URL'i (ör. 'E0', '2324') — ayna kapsamıyorsa None."""
    d = MIRROR_DIRS.get(league_code)
    return MIRROR_URL.format(dir=d, season=season) if d else None

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
    "B1": 144,   # Jupiler Pro League
    "SWE": 113,  # Allsvenskan
    "NOR": 103,  # Eliteserien
    "DNK": 119,  # Superliga
    "POL": 106,  # Ekstraklasa
    "AUT": 218,  # Österreichische Bundesliga
    "SWZ": 207,  # Swiss Super League
}

# --------------------------------------------------------------------------- #
# Model varsayılanları
# --------------------------------------------------------------------------- #
DEFAULT_HALF_LIFE_DAYS = 180.0   # zaman ağırlığı üstel azalma yarı-ömrü
DEFAULT_MAX_GOALS = 10           # skor matrisi boyutu (0..MAX_GOALS)
DEFAULT_SEASONS_BACK = 4         # kaç sezon otomatik indirilsin

# HTTP
HTTP_TIMEOUT = 30
HTTP_RETRIES = 3   # geçici ağ hatalarında yeniden deneme sayısı (1s/2s/4s bekleyerek)
USER_AGENT = "FootballPredictor/0.1 (+educational, statistical modelling)"
