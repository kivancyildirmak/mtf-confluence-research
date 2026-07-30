"""API-Football istemcisi testleri.

Canlı API çağrısı YAPILMAZ: `requests.get` sahte yanıtlarla değiştirilir.
Böylece ayrıştırma, hata yolları ve kota okuma ağ olmadan doğrulanır.
"""
from datetime import date

import pytest

from fpredict import api_football as af


# --------------------------------------------------------------------------- #
# Sahte HTTP altyapısı
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None, bad_json=False):
        self._payload = payload if payload is not None else {"response": []}
        self.status_code = status
        self.headers = headers or {}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


def patch_get(monkeypatch, response, capture=None):
    """requests.get'i sabit bir yanıtla değiştirir; çağrıyı capture'a yazar."""
    import requests

    def fake_get(url, params=None, headers=None, timeout=None):
        if capture is not None:
            capture.append({"url": url, "params": params, "headers": headers})
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests, "get", fake_get)


def _fixture(home, away, hg, ag, iso="2026-04-12T18:00:00+00:00", status="FT"):
    return {
        "fixture": {"date": iso, "status": {"short": status}},
        "teams": {"home": {"name": home}, "away": {"name": away}},
        "goals": {"home": hg, "away": ag},
    }


# --------------------------------------------------------------------------- #
# İstemci temelleri
# --------------------------------------------------------------------------- #
def test_client_requires_key():
    with pytest.raises(af.APIFootballError):
        af.APIFootballClient("")
    with pytest.raises(af.APIFootballError):
        af.APIFootballClient("   ")


def test_sends_api_key_header(monkeypatch):
    cap = []
    patch_get(monkeypatch, FakeResponse({"response": []}), cap)
    af.APIFootballClient("SECRET").fetch_fixtures(103, 2026, "NOR")
    assert cap[0]["headers"]["x-apisports-key"] == "SECRET"
    assert cap[0]["params"] == {"league": 103, "season": 2026}


def test_quota_read_from_headers(monkeypatch):
    patch_get(monkeypatch, FakeResponse(
        {"response": []},
        headers={"x-ratelimit-requests-current": "7",
                 "x-ratelimit-requests-limit": "100"},
    ))
    c = af.APIFootballClient("k")
    c.fetch_fixtures(103, 2026, "NOR")
    assert c.last_quota.used == 7
    assert c.last_quota.limit == 100
    assert c.last_quota.remaining == 93
    assert "93" in c.last_quota.describe()


# --------------------------------------------------------------------------- #
# Hata yolları
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status,fragment", [
    (401, "geçersiz"),
    (403, "geçersiz"),
    (429, "kota"),
    (500, "HTTP 500"),
])
def test_http_error_messages(monkeypatch, status, fragment):
    patch_get(monkeypatch, FakeResponse({}, status=status))
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("k").status()
    assert fragment.lower() in str(exc.value).lower()


def test_api_level_errors_raise(monkeypatch):
    """API 200 döndürüp gövdede hata bildirebilir."""
    patch_get(monkeypatch, FakeResponse({"errors": {"token": "invalid"}}))
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("k").fetch_fixtures(1, 2026, "X")
    assert "token" in str(exc.value)


def test_empty_errors_list_is_not_an_error(monkeypatch):
    """API boş liste/dict gönderdiğinde hata sayılmamalı."""
    patch_get(monkeypatch, FakeResponse({"errors": [], "response": []}))
    assert af.APIFootballClient("k").fetch_fixtures(1, 2026, "X") == []


def test_bad_json_raises(monkeypatch):
    patch_get(monkeypatch, FakeResponse(bad_json=True))
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("k").status()
    assert "JSON" in str(exc.value)


def test_network_error_is_wrapped(monkeypatch):
    import requests
    patch_get(monkeypatch, requests.exceptions.ConnectTimeout("timed out"))
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("k").status()
    assert "erişilemedi" in str(exc.value)


# --------------------------------------------------------------------------- #
# Fixture ayrıştırma
# --------------------------------------------------------------------------- #
def test_fetch_fixtures_normalizes_rows(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        _fixture("Bodo/Glimt", "Lillestrom", 3, 1),
    ]}))
    rows = af.APIFootballClient("k").fetch_fixtures(103, 2026, "NOR")
    assert len(rows) == 1
    r = rows[0]
    assert r["league"] == "NOR"
    assert r["date"] == "2026-04-12"          # ISO8601 -> tarih
    assert r["home_team"] == "Bodo/Glimt"
    assert r["home_goals"] == 3 and r["away_goals"] == 1
    assert r["result"] == "H"
    assert r["season"] == "2026"
    assert r["match_uid"]                      # tekilleştirme için gerekli


def test_fetch_fixtures_skips_unfinished(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        _fixture("A", "B", 1, 0, status="FT"),
        _fixture("C", "D", None, None, status="NS"),    # oynanmadı
        _fixture("E", "F", None, None, status="PST"),   # ertelendi
        _fixture("G", "H", 2, 2, status="AET"),         # uzatma -> sayılır
    ]}))
    rows = af.APIFootballClient("k").fetch_fixtures(1, 2026, "X")
    assert [r["home_team"] for r in rows] == ["A", "G"]


def test_fetch_fixtures_skips_missing_goals(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        {"fixture": {"date": "2026-04-12T00:00:00+00:00", "status": {"short": "FT"}},
         "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
         "goals": {"home": None, "away": 1}},
    ]}))
    assert af.APIFootballClient("k").fetch_fixtures(1, 2026, "X") == []


def test_fetch_fixtures_result_values(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        _fixture("A", "B", 0, 2), _fixture("C", "D", 1, 1),
    ]}))
    rows = af.APIFootballClient("k").fetch_fixtures(1, 2026, "X")
    assert [r["result"] for r in rows] == ["A", "D"]


def test_fetch_fixtures_anchors_names_to_cache(monkeypatch):
    """API adları cache'teki yazıma bağlanmalı ki takım bölünmesin."""
    patch_get(monkeypatch, FakeResponse({"response": [
        _fixture("Valerenga Fotball", "HamKam", 2, 1),
    ]}))
    rows = af.APIFootballClient("k").fetch_fixtures(
        103, 2026, "NOR", known_teams=["Valerenga", "Ham-Kam", "Molde"],
    )
    assert rows[0]["home_team"] == "Valerenga"
    assert rows[0]["away_team"] == "Ham-Kam"


def test_unknown_team_keeps_api_name(monkeypatch):
    """Cache'te karşılığı olmayan (ör. yeni yükselen) takım korunmalı."""
    patch_get(monkeypatch, FakeResponse({"response": [
        _fixture("Yeni Takim FK", "Molde", 0, 3),
    ]}))
    rows = af.APIFootballClient("k").fetch_fixtures(
        103, 2026, "NOR", known_teams=["Molde", "Brann"],
    )
    assert rows[0]["home_team"] == "Yeni Takim FK"


# --------------------------------------------------------------------------- #
# Takım / lig arama
# --------------------------------------------------------------------------- #
def test_find_team_id_tolerates_spelling(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        {"team": {"id": 327, "name": "Ham-Kam"}},
        {"team": {"id": 331, "name": "Molde"}},
    ]}))
    assert af.APIFootballClient("k").find_team_id(103, 2026, "HamKam") == 327


def test_find_team_id_returns_none_when_absent(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        {"team": {"id": 1, "name": "Molde"}},
    ]}))
    assert af.APIFootballClient("k").find_team_id(103, 2026, "Real Madrid") is None


def test_search_leagues_shape(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        {"league": {"id": 103, "name": "Eliteserien", "type": "League"},
         "country": {"name": "Norway"},
         "seasons": [{"year": 2025}, {"year": 2026}]},
    ]}))
    out = af.APIFootballClient("k").search_leagues("Elite")
    assert out[0]["id"] == 103
    assert out[0]["country"] == "Norway"
    assert out[0]["latest_season"] == 2026


def test_search_leagues_requires_min_length():
    with pytest.raises(af.APIFootballError):
        af.APIFootballClient("k").search_leagues("ab")


def test_fetch_injuries(monkeypatch):
    patch_get(monkeypatch, FakeResponse({"response": [
        {"player": {"name": "A. Player", "type": "Missing Fixture",
                    "reason": "Knee Injury"}},
    ]}))
    out = af.APIFootballClient("k").fetch_injuries(327, 2026)
    assert out == [{"player": "A. Player", "type": "Missing Fixture",
                    "reason": "Knee Injury"}]


# --------------------------------------------------------------------------- #
# Sezon yılı hesabı
# --------------------------------------------------------------------------- #
def test_api_seasons_calendar_year_league():
    """İskandinav ligleri takvim yılıdır: Temmuz 2026 -> 2026 sezonu."""
    assert af.api_seasons("NOR", 2, today=date(2026, 7, 30)) == [2026, 2025]
    # Mart ayında da içinde bulunulan yıl geçerlidir
    assert af.api_seasons("SWE", 2, today=date(2026, 3, 10)) == [2026, 2025]


def test_api_seasons_autumn_spring_league():
    """Premier Lig: Temmuz'dan sonra yeni sezon, öncesinde bir önceki."""
    assert af.api_seasons("E0", 2, today=date(2026, 7, 30)) == [2026, 2025]
    assert af.api_seasons("E0", 2, today=date(2026, 3, 10)) == [2025, 2024]


def test_api_seasons_denmark_is_split_year():
    """Danimarka Superliga sonbahar-ilkbahar oynanır (takvim yılı DEĞİL)."""
    assert af.api_seasons("DNK", 1, today=date(2026, 3, 10)) == [2025]


# --------------------------------------------------------------------------- #
# update_from_api: uçtan uca akış (sahte API -> SQLite -> model)
# --------------------------------------------------------------------------- #
def test_update_from_api_writes_to_db(monkeypatch, tmp_path):
    from fpredict import data_fetch, db, squad_adjust

    monkeypatch.setattr(squad_adjust, "load_league_id", lambda k: 103)
    patch_get(monkeypatch, FakeResponse({"response": [
        _fixture("Molde", "Brann", 2, 0, iso="2026-05-01T18:00:00+00:00"),
        _fixture("Brann", "Molde", 1, 1, iso="2026-06-01T18:00:00+00:00"),
    ]}))

    dbf = tmp_path / "t.sqlite"
    res = data_fetch.update_from_api("NOR", "key", seasons_back=1, db_path=dbf)

    assert res["inserted"] == 2
    assert res["sources_used"] == ["API-Football"]
    assert res["seasons_ok"]
    stored = db.load_matches("NOR", db_path=dbf)
    assert len(stored) == 2
    assert set(stored["home_team"]) == {"Molde", "Brann"}


def test_update_from_api_reports_zero_results(monkeypatch, tmp_path):
    """Boş yanıt, yanlış lig id'sine işaret eder — kullanıcıya söylenmeli."""
    from fpredict import data_fetch, squad_adjust

    monkeypatch.setattr(squad_adjust, "load_league_id", lambda k: 999999)
    patch_get(monkeypatch, FakeResponse({"response": []}))

    res = data_fetch.update_from_api("NOR", "key", seasons_back=1,
                                     db_path=tmp_path / "t.sqlite")
    assert res["inserted"] == 0
    assert res["seasons_failed"]
    assert "lig id" in " ".join(res["errors"]).lower()


def test_update_from_api_without_league_id_raises(monkeypatch, tmp_path):
    from fpredict import data_fetch, squad_adjust

    monkeypatch.setattr(squad_adjust, "load_league_id", lambda k: None)
    with pytest.raises(data_fetch.DataFetchError) as exc:
        data_fetch.update_from_api("NOR", "key", db_path=tmp_path / "t.sqlite")
    assert "lig id" in str(exc.value).lower()


def test_update_from_api_survives_partial_failure(monkeypatch, tmp_path):
    """Bir sezon patlarsa diğerleri yine de kaydedilmeli."""
    from fpredict import data_fetch, squad_adjust
    import requests

    monkeypatch.setattr(squad_adjust, "load_league_id", lambda k: 103)
    calls = {"n": 0}

    def flaky(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse({"response": [
                _fixture("Molde", "Brann", 2, 0, iso="2026-05-01T00:00:00+00:00")]})
        raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr(requests, "get", flaky)
    res = data_fetch.update_from_api("NOR", "key", seasons_back=2,
                                     db_path=tmp_path / "t.sqlite")
    assert res["inserted"] == 1
    assert len(res["seasons_ok"]) == 1
    assert len(res["seasons_failed"]) == 1


# --------------------------------------------------------------------------- #
# Ayar kalıcılığı: lig ID override
# --------------------------------------------------------------------------- #
def test_league_id_override_persists(tmp_path, monkeypatch):
    """Kullanıcının doğruladığı ID, yerleşik varsayılanı geçersiz kılmalı."""
    from fpredict import config, squad_adjust

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

    # Önce yerleşik varsayılan
    assert squad_adjust.load_league_id("NOR") == config.APIFOOTBALL_LEAGUE_IDS["NOR"]

    squad_adjust.save_league_id("NOR", 999)
    assert squad_adjust.load_league_id("NOR") == 999
    # Diğer ligler etkilenmemeli
    assert squad_adjust.load_league_id("SWE") == config.APIFOOTBALL_LEAGUE_IDS["SWE"]


def test_api_key_and_league_ids_coexist(tmp_path, monkeypatch):
    """Anahtar kaydı lig ID'lerini silmemeli (aynı settings.json dosyası)."""
    from fpredict import config, squad_adjust

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

    squad_adjust.save_league_id("NOR", 103)
    squad_adjust.save_api_key("SECRET")
    assert squad_adjust.load_api_key() == "SECRET"
    assert squad_adjust.load_league_id("NOR") == 103

    squad_adjust.save_league_id("SWE", 113)
    assert squad_adjust.load_api_key() == "SECRET"


def test_all_leagues_have_api_id():
    """Her lig API ile güncellenebilmeli (id eksikse kaynak kullanılamaz)."""
    from fpredict import config
    missing = [k for k in config.LEAGUES if k not in config.APIFOOTBALL_LEAGUE_IDS]
    assert not missing, f"API id tanımsız ligler: {missing}"


# --------------------------------------------------------------------------- #
# İki kanal (api-sports.io doğrudan / RapidAPI)
# --------------------------------------------------------------------------- #
def test_direct_provider_headers_and_url(monkeypatch):
    cap = []
    patch_get(monkeypatch, FakeResponse({"response": []}), cap)
    af.APIFootballClient("K", provider="direct").fetch_fixtures(1, 2026, "X")
    assert cap[0]["url"].startswith("https://v3.football.api-sports.io/")
    assert cap[0]["headers"] == {"x-apisports-key": "K"}


def test_rapidapi_provider_headers_and_url(monkeypatch):
    cap = []
    patch_get(monkeypatch, FakeResponse({"response": []}), cap)
    af.APIFootballClient("K", provider="rapidapi").fetch_fixtures(1, 2026, "X")
    assert cap[0]["url"].startswith("https://api-football-v1.p.rapidapi.com/v3/")
    assert cap[0]["headers"]["x-rapidapi-key"] == "K"
    assert cap[0]["headers"]["x-rapidapi-host"] == "api-football-v1.p.rapidapi.com"


def test_unknown_provider_rejected():
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("K", provider="nope")
    assert "sağlayıcı" in str(exc.value).lower()


def test_rapidapi_quota_from_remaining_header(monkeypatch):
    """RapidAPI 'kalan' bildirir, api-sports 'kullanılan' — ikisi de okunmalı."""
    patch_get(monkeypatch, FakeResponse(
        {"response": []},
        headers={"x-ratelimit-requests-limit": "100",
                 "x-ratelimit-requests-remaining": "88"},
    ))
    c = af.APIFootballClient("K", provider="rapidapi")
    c.fetch_fixtures(1, 2026, "X")
    assert c.last_quota.limit == 100
    assert c.last_quota.used == 12
    assert c.last_quota.remaining == 88


def test_provider_error_message_names_channel(monkeypatch):
    import requests
    patch_get(monkeypatch, requests.exceptions.ConnectTimeout("timed out"))
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("K", provider="rapidapi").status()
    assert "RapidAPI" in str(exc.value)


def test_provider_setting_roundtrip(tmp_path, monkeypatch):
    from fpredict import config, squad_adjust

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")

    assert squad_adjust.load_api_provider() == "direct"   # varsayılan
    squad_adjust.save_api_provider("rapidapi")
    assert squad_adjust.load_api_provider() == "rapidapi"
    # anahtar kaydı kanalı silmemeli
    squad_adjust.save_api_key("K")
    assert squad_adjust.load_api_provider() == "rapidapi"


def test_403_message_suggests_channel_mismatch(monkeypatch):
    """403'ün en sık sebebi kanal/anahtar uyuşmazlığı — mesaj bunu söylemeli."""
    patch_get(monkeypatch, FakeResponse({}, status=403))
    with pytest.raises(af.APIFootballError) as exc:
        af.APIFootballClient("k", provider="rapidapi").status()
    msg = str(exc.value)
    assert "uyuşmazlığ" in msg.lower()
    assert "RapidAPI" in msg                       # seçili kanal
    assert PROVIDERS_DIRECT_SIGNUP in msg          # önerilen diğer kanal

PROVIDERS_DIRECT_SIGNUP = af.PROVIDERS["direct"]["signup"]
