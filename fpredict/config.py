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
    # Takvim yılı ligi mi? (İskandinavya, MLS, Brezilya: Mart-Kasım tek yıl)
    # False ise sonbahar-ilkbahar ligi (2025-26 gibi) demektir.
    calendar_year: bool = False


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
    "SWE": LeagueInfo("İsveç Allsvenskan", "SWE", "extra", "İsveç", calendar_year=True),
    "NOR": LeagueInfo("Norveç Eliteserien", "NOR", "extra", "Norveç", calendar_year=True),
    "DNK": LeagueInfo("Danimarka Superliga", "DNK", "extra", "Danimarka"),
    "FIN": LeagueInfo("Finlandiya Veikkausliiga", "FIN", "extra", "Finlandiya", calendar_year=True),
    "POL": LeagueInfo("Polonya Ekstraklasa", "POL", "extra", "Polonya"),
    "AUT": LeagueInfo("Avusturya Bundesliga", "AUT", "extra", "Avusturya"),
    "SWZ": LeagueInfo("İsviçre Super Lig", "SWZ", "extra", "İsviçre"),
    "IRL": LeagueInfo("İrlanda Premier Division", "IRL", "extra", "İrlanda", calendar_year=True),
    "ROU": LeagueInfo("Romanya Liga 1", "ROU", "extra", "Romanya"),
    "RUS": LeagueInfo("Rusya Premier Lig", "RUS", "extra", "Rusya"),
    "USA": LeagueInfo("ABD MLS", "USA", "extra", "ABD", calendar_year=True),
    "MEX": LeagueInfo("Meksika Liga MX", "MEX", "extra", "Meksika"),
    "BRA": LeagueInfo("Brezilya Serie A", "BRA", "extra", "Brezilya", calendar_year=True),
    "ARG": LeagueInfo("Arjantin Primera Division", "ARG", "extra", "Arjantin"),
    "JPN": LeagueInfo("Japonya J1 Lig", "JPN", "extra", "Japonya", calendar_year=True),
    "CHN": LeagueInfo("Çin Super Lig", "CHN", "extra", "Çin", calendar_year=True),
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


def key_for_code(code: str) -> str:
    """Cache'te saklanan lig kodundan LEAGUES anahtarını bulur.

    Şu an tüm ligler için anahtar == kod, ancak arayüz cache'ten kodu okuyup
    LEAGUES'e anahtarla eriştiği için bu dönüşümü varsayım yerine açık bir
    fonksiyona bağlıyoruz.
    """
    if code in LEAGUES:
        return code
    for key, info in LEAGUES.items():
        if info.code == code:
            return key
    return code


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


# --------------------------------------------------------------------------- #
# Arşiv kaynağı (tüm ligler, tek dosya — ama güncel değil)
# --------------------------------------------------------------------------- #
# football-data.co.uk'tan türetilmiş, 2000-2025 arasını kapsayan birleşik veri
# seti. GitHub üzerinde barındığı için football-data.co.uk engelliyken de
# erişilebilir. 38 lig içerir (İskandinavya, Polonya, Belçika dahil).
#
# ÖNEMLİ: Bu bir ARŞİVDİR, canlı kaynak değildir. Son maç tarihleri ligden lige
# değişir (büyük Avrupa ligleri ~2025 Mayıs, İskandinav ligleri ~2024 Aralık).
# Güncel sezon verisi için asıl kaynağa veya API-Football'a ihtiyaç vardır.
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025"
    "/main/data/Matches.csv"
)
ARCHIVE_CACHE = APP_DIR / "archive_matches.csv"
ARCHIVE_MAX_AGE_DAYS = 14   # yerel arşiv kopyası bu kadar gün sonra yenilenir

# Bizim lig kodumuz -> arşivdeki 'Division' kodu (bazıları farklı yazılır)
ARCHIVE_DIVISIONS: dict[str, str] = {
    "E0": "E0", "E1": "E1", "SC0": "SC0",
    "SP1": "SP1", "SP2": "SP2",
    "D1": "D1", "D2": "D2",
    "I1": "I1", "I2": "I2",
    "F1": "F1", "F2": "F2",
    "N1": "N1", "B1": "B1", "P1": "P1", "G1": "G1", "T1": "T1",
    "SWE": "SWE", "NOR": "NOR", "DNK": "DEN", "FIN": "FIN",
    "POL": "POL", "AUT": "AUT", "SWZ": "SUI", "IRL": "IRL",
    "ROU": "ROM", "RUS": "RUS", "USA": "USA", "MEX": "MEX",
    "BRA": "BRA", "ARG": "ARG", "JPN": "JAP", "CHN": "CHN",
}


def archive_division(league_key: str) -> str | None:
    """Lig anahtarının arşivdeki karşılığı (yoksa None)."""
    return ARCHIVE_DIVISIONS.get(league_key)

# football-data.org lig id eşlemesi (sakatlık/kadro API'si opsiyonel kullanım)
# API-Football (api-sports.io) league id'leri
# NOT: Bu id'ler API-Football'un yayımladığı lig kimlikleridir. Yanlış/eskimiş
# olma ihtimaline karşı arayüzde "Lig ID ara/doğrula" aracı vardır; kullanıcının
# bulduğu id settings.json içine yazılır ve buradaki varsayılanı geçersiz kılar.
APIFOOTBALL_LEAGUE_IDS: dict[str, int] = {
    "T1": 203,   # Süper Lig
    "E0": 39,    # Premier League
    "E1": 40,    # Championship
    "SC0": 179,  # Scottish Premiership
    "SP1": 140,  # La Liga
    "SP2": 141,  # La Liga 2
    "D1": 78,    # Bundesliga
    "D2": 79,    # 2. Bundesliga
    "I1": 135,   # Serie A
    "I2": 136,   # Serie B
    "F1": 61,    # Ligue 1
    "F2": 62,    # Ligue 2
    "N1": 88,    # Eredivisie
    "P1": 94,    # Primeira Liga
    "G1": 197,   # Super League Greece
    "B1": 144,   # Jupiler Pro League
    "SWE": 113,  # Allsvenskan
    "NOR": 103,  # Eliteserien
    "DNK": 119,  # Superliga
    "FIN": 244,  # Veikkausliiga
    "POL": 106,  # Ekstraklasa
    "AUT": 218,  # Österreichische Bundesliga
    "SWZ": 207,  # Swiss Super League
    "IRL": 357,  # Premier Division
    "ROU": 283,  # Liga I
    "RUS": 235,  # Premier League
    "USA": 253,  # MLS
    "MEX": 262,  # Liga MX
    "BRA": 71,   # Serie A
    "ARG": 128,  # Liga Profesional
    "JPN": 98,   # J1 League
    "CHN": 169,  # Super League
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
