"""API-Football (api-sports.io) istemcisi — güncel maç sonuçları ve sakatlıklar.

Neden gerekli
-------------
football-data.co.uk bazı ağlarda erişilemez; arşiv kaynağı ise canlı değildir
(İskandinav ligleri ~Aralık 2024'te biter). API-Football ücretsiz kotayla
(günde 100 istek) **güncel sezon** verisi verir ve tüm ligleri kapsar.

Kullanılan uç noktalar
----------------------
    GET /status                          -> hesap/kota bilgisi
    GET /fixtures?league={id}&season={y}  -> sezonun tüm maçları (1 istek)
    GET /teams?league={id}&season={y}     -> takım id'leri
    GET /injuries?team={id}&season={y}    -> sakat/cezalı oyuncular

Kota
----
Bir sezonun tüm maçları TEK istekte gelir; 4 lig x 2 sezon = 8 istek. Ücretsiz
kota (100/gün) bunun için fazlasıyla yeterlidir. İstemci her yanıttan kalan
kotayı okur ve `last_quota` içinde tutar.

NOT: Bu modül, geliştirme ortamından canlı doğrulanamamıştır (api-sports.io o
ortamda engelliydi). Bu yüzden hata yolları savunmacı yazılmış ve sahte
yanıtlarla test edilmiştir; ilk canlı çalıştırmada `status()` ile bağlantıyı
doğrulamak önerilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from . import config, db
from .name_matching import best_match

BASE_URL = "https://v3.football.api-sports.io"

# Yalnızca tamamlanmış maçlar modele girer (NS = başlamadı, PST = ertelendi …)
FINISHED_STATUSES = {"FT", "AET", "PEN"}


class APIFootballError(RuntimeError):
    """Kullanıcıya gösterilebilir API hatası."""


@dataclass
class APIQuota:
    """Yanıt başlıklarından okunan günlük kota durumu."""
    used: int | None = None
    limit: int | None = None

    @property
    def remaining(self) -> int | None:
        if self.used is None or self.limit is None:
            return None
        return max(0, self.limit - self.used)

    def describe(self) -> str:
        if self.limit is None:
            return "kota bilgisi yok"
        return f"{self.used or 0}/{self.limit} istek kullanıldı (kalan: {self.remaining})"


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_name(name: str, known_teams=None) -> str:
    """API'den gelen takım adını, cache'te zaten var olan yazıma bağlar.

    Aynı takım iki kaynakta farklı yazılırsa ('Valerenga' / 'Valerenga Fotball')
    ayrı takım sayılır ve maç geçmişi bölünür. Bu yüzden API adları, cache'teki
    mevcut adlara *yüksek eşikle* çapalanır; eşleşme yoksa API adı korunur.
    """
    if not known_teams:
        return name
    if name in known_teams:
        return name
    match, _score = best_match(name, list(known_teams), threshold=0.8)
    return match or name


class APIFootballClient:
    """API-Football v3 için ince istemci."""

    def __init__(self, api_key: str, timeout: int = config.HTTP_TIMEOUT):
        if not api_key or not str(api_key).strip():
            raise APIFootballError("API anahtarı girilmemiş.")
        self.api_key = str(api_key).strip()
        self.timeout = timeout
        self.last_quota: APIQuota | None = None

    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: dict | None = None) -> dict:
        """Tek bir GET isteği; HTTP ve API-içi hataları anlamlı mesaja çevirir."""
        import requests

        from .data_fetch import classify_network_error

        url = f"{BASE_URL}/{path.lstrip('/')}"
        try:
            resp = requests.get(
                url,
                params=params or {},
                headers={"x-apisports-key": self.api_key},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise APIFootballError(
                f"API'ye erişilemedi: {classify_network_error(exc)}"
            ) from exc

        # Kota başlıkları (varsa) her yanıtta güncellenir
        self.last_quota = APIQuota(
            used=_int(resp.headers.get("x-ratelimit-requests-current")),
            limit=_int(resp.headers.get("x-ratelimit-requests-limit")),
        )

        if resp.status_code in (401, 403):
            raise APIFootballError(
                f"API anahtarı geçersiz veya yetkisiz (HTTP {resp.status_code}). "
                "Anahtarı Kadro/Sakatlık ekranından kontrol edin."
            )
        if resp.status_code == 429:
            raise APIFootballError(
                "Günlük istek kotanız doldu (HTTP 429). Yarın tekrar deneyin "
                "veya planınızı yükseltin."
            )
        if resp.status_code >= 400:
            raise APIFootballError(f"API HTTP {resp.status_code} döndü.")

        try:
            data = resp.json()
        except ValueError as exc:
            raise APIFootballError("API yanıtı okunamadı (geçersiz JSON).") from exc

        # API 200 döndürüp gövdede hata bildirebilir
        errors = data.get("errors")
        if errors:
            if isinstance(errors, dict) and errors:
                raise APIFootballError(f"API hatası: {errors}")
            if isinstance(errors, list) and errors:
                raise APIFootballError(f"API hatası: {errors}")
        return data

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        """Hesap/kota bilgisi — anahtarı doğrulamak için ucuz bir çağrı."""
        data = self._get("status")
        resp = data.get("response") or {}
        sub = resp.get("subscription") or {}
        reqs = resp.get("requests") or {}
        # /status kendi gövdesinde de kota verir; başlıklardan daha güvenilirdir
        if reqs:
            self.last_quota = APIQuota(
                used=_int(reqs.get("current")), limit=_int(reqs.get("limit_day"))
            )
        return {
            "account": (resp.get("account") or {}).get("email", ""),
            "plan": sub.get("plan", ""),
            "active": sub.get("active"),
            "quota": self.last_quota,
        }

    def search_leagues(self, name: str) -> list[dict]:
        """Lig adına göre arama — lig id'sini doğrulamak/bulmak için."""
        if not name or len(name) < 3:
            raise APIFootballError("Arama için en az 3 karakter girin.")
        data = self._get("leagues", {"search": name})
        out = []
        for item in data.get("response", []):
            lg = item.get("league") or {}
            country = (item.get("country") or {}).get("name", "")
            seasons = [s.get("year") for s in (item.get("seasons") or [])]
            out.append({
                "id": lg.get("id"),
                "name": lg.get("name"),
                "type": lg.get("type"),
                "country": country,
                "latest_season": max([s for s in seasons if s], default=None),
            })
        return out

    def list_teams(self, league_id: int, season: int) -> list[dict]:
        """Ligdeki takımlar (id + ad) — sakatlık sorgusu için gerekir."""
        data = self._get("teams", {"league": league_id, "season": season})
        return [
            {"id": (t.get("team") or {}).get("id"),
             "name": (t.get("team") or {}).get("name")}
            for t in data.get("response", [])
        ]

    def find_team_id(self, league_id: int, season: int, team_name: str):
        """Takım adından id bulur (yazım farklarını tolere ederek)."""
        teams = self.list_teams(league_id, season)
        names = [t["name"] for t in teams if t.get("name")]
        match, _ = best_match(team_name, names, threshold=0.75)
        if not match:
            return None
        for t in teams:
            if t["name"] == match:
                return t["id"]
        return None

    # ------------------------------------------------------------------ #
    def fetch_fixtures(
        self,
        league_id: int,
        season: int,
        league_code: str,
        known_teams=None,
    ) -> list[dict]:
        """Bir sezonun tamamlanmış maçlarını normalize kayıt listesine çevirir.

        Args:
            league_id: API-Football lig id'si.
            season: sezon yılı (API'de sezonun BAŞLADIĞI yıl).
            league_code: bizim cache'teki lig kodu (ör. 'NOR').
            known_teams: cache'te zaten var olan takım adları — API adlarını
                bunlara çapalamak için (aynı takımın bölünmesini önler).

        Returns:
            db.upsert_matches ile uyumlu satır listesi. Oynanmamış/ertelenmiş
            maçlar atlanır (API bunları da döndürür).
        """
        data = self._get("fixtures", {"league": league_id, "season": season})
        rows: list[dict] = []
        for item in data.get("response", []):
            fixture = item.get("fixture") or {}
            status = ((fixture.get("status") or {}).get("short") or "").upper()
            if status not in FINISHED_STATUSES:
                continue

            raw_date = fixture.get("date") or ""
            iso = raw_date[:10]  # '2026-04-12T18:00:00+00:00' -> '2026-04-12'
            if len(iso) != 10:
                continue

            teams = item.get("teams") or {}
            home = ((teams.get("home") or {}).get("name") or "").strip()
            away = ((teams.get("away") or {}).get("name") or "").strip()
            if not home or not away:
                continue

            goals = item.get("goals") or {}
            hg, ag = _int(goals.get("home")), _int(goals.get("away"))
            if hg is None or ag is None:
                continue

            home = _canonical_name(home, known_teams)
            away = _canonical_name(away, known_teams)

            rows.append({
                "match_uid": db.make_uid(league_code, iso, home, away),
                "league": league_code,
                "season": str(season),
                "date": iso,
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "result": "H" if hg > ag else "A" if ag > hg else "D",
                # API-Football ücretsiz planında bahis oranları yoktur
                "odds_h": None, "odds_d": None, "odds_a": None,
                "odds_over25": None, "odds_under25": None,
            })
        return rows

    def fetch_injuries(self, team_id: int, season: int) -> list[dict]:
        """Bir takımın sakat/cezalı oyuncuları."""
        data = self._get("injuries", {"team": team_id, "season": season})
        out = []
        for item in data.get("response", []):
            player = item.get("player") or {}
            out.append({
                "player": player.get("name", "?"),
                "type": player.get("type", ""),
                "reason": player.get("reason", ""),
            })
        return out


# --------------------------------------------------------------------------- #
# Sezon yılı hesabı
# --------------------------------------------------------------------------- #
def api_seasons(league_key: str, n: int, today: date | None = None) -> list[int]:
    """API-Football için indirilecek sezon yıllarını döndürür.

    İki takvim vardır:
      * takvim yılı ligleri (İsveç, Norveç, Finlandiya, MLS, Brezilya…):
        sezon = içinde bulunulan yıl.
      * sonbahar-ilkbahar ligleri (Premier Lig, Süper Lig, Danimarka…):
        sezon = sezonun BAŞLADIĞI yıl (Temmuz'dan önce bir önceki yıl).
    """
    today = today or date.today()
    info = config.LEAGUES.get(league_key)
    if info is not None and getattr(info, "calendar_year", False):
        start = today.year
    else:
        start = today.year if today.month >= 7 else today.year - 1
    return [start - i for i in range(n)]
