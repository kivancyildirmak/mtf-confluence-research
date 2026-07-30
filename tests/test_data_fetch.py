"""Veri çekme / CSV ayrıştırma testleri (ağ gerektirmez)."""
from datetime import date
from pathlib import Path

import pytest

from fpredict import data_fetch as dfetch


SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "T1_sample.csv"
SAMPLE_EXTRA = Path(__file__).resolve().parent.parent / "sample_data" / "NOR_sample.csv"


def test_season_code():
    assert dfetch.season_code(2023) == "2324"
    assert dfetch.season_code(1999) == "9900"
    assert dfetch.season_code(2009) == "0910"


def test_recent_seasons_summer_boundary():
    # Ağustos: içinde bulunulan yıl sezon başlangıcı
    assert dfetch.recent_seasons(3, today=date(2024, 8, 20)) == [2024, 2023, 2022]
    # Şubat: bir önceki yıl hâlâ aktif sezon
    assert dfetch.recent_seasons(2, today=date(2024, 2, 10)) == [2023, 2022]


def test_csv_url_pattern():
    url = dfetch.csv_url("E0", 2023)
    assert url.endswith("/2324/E0.csv")
    assert url.startswith("https://")


def test_parse_sample_csv():
    raw = SAMPLE.read_bytes()
    rows = dfetch.parse_csv_bytes(raw, "T1", season="2223")
    assert len(rows) > 100
    first = rows[0]
    assert first["league"] == "T1"
    assert first["home_team"]
    assert first["away_team"]
    assert isinstance(first["home_goals"], int)
    assert first["result"] in {"H", "D", "A"}
    # oran sütunları okunmuş olmalı
    assert first["odds_h"] is not None


def test_parse_date_formats():
    assert dfetch._parse_date("05/08/2022") == "2022-08-05"
    assert dfetch._parse_date("5/8/22") == "2022-08-05"
    assert dfetch._parse_date("garbage") is None


def test_parse_date_iso_not_day_month_swapped():
    """Regresyon: ISO tarihlerde gün/ay takası olmamalı.

    Esnek ayrıştırıcı dayfirst=True ile '2023-09-01' değerini 9 Ocak olarak
    okuyordu (YYYY-DD-MM). Hem gün hem ay <= 12 olduğunda hata sessizdi ve
    zaman ağırlığını/backtest sıralamasını bozuyordu.
    """
    assert dfetch._parse_date("2023-09-01") == "2023-09-01"   # 1 Eylül
    assert dfetch._parse_date("2026-04-12") == "2026-04-12"   # 12 Nisan
    assert dfetch._parse_date("2025-12-04") == "2025-12-04"   # 4 Aralık
    assert dfetch._parse_date("2025-08-15") == "2025-08-15"   # tek anlamlı


def test_parse_iso_dated_csv_keeps_chronology():
    """Ayna kaynağı ISO tarih kullanır; sezon aralığı bozulmamalı."""
    csv = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "2025-08-15,Liverpool,Bournemouth,4,2,H\n"
        "2025-09-01,Arsenal,Chelsea,1,0,H\n"
        "2026-04-12,Man City,Everton,2,2,D\n"
    )
    rows = dfetch.parse_csv_bytes(csv, "E0")
    dates = [r["date"] for r in rows]
    assert dates == ["2025-08-15", "2025-09-01", "2026-04-12"]
    assert dates == sorted(dates)  # kronoloji korunmalı


def test_parse_missing_columns_raises():
    bad = "Foo,Bar\n1,2\n"
    with pytest.raises(dfetch.DataFetchError):
        dfetch.parse_csv_bytes(bad, "T1")


def test_parse_skips_unplayed_matches():
    csv = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "T1,05/08/2022,A,B,2,1,H\n"
        "T1,06/08/2022,C,D,,,\n"  # oynanmamış
    )
    rows = dfetch.parse_csv_bytes(csv, "T1")
    assert len(rows) == 1
    assert rows[0]["home_team"] == "A"


# --------------------------------------------------------------------------- #
# 'extra' biçim (İskandinavya, Polonya vb. — /new/{KOD}.csv)
# --------------------------------------------------------------------------- #
def test_extra_csv_url_pattern():
    url = dfetch.extra_csv_url("NOR")
    assert url.endswith("/new/NOR.csv")
    assert url.startswith("https://")


def test_parse_extra_sample_csv():
    rows = dfetch.parse_extra_csv_bytes(SAMPLE_EXTRA.read_bytes(), "NOR")
    assert len(rows) > 100
    first = rows[0]
    assert first["league"] == "NOR"
    assert first["home_team"] and first["away_team"]
    assert isinstance(first["home_goals"], int)
    assert first["result"] in {"H", "D", "A"}
    # extra biçimde Bet365 yok; Avg/Pinnacle sütunlarından okunmalı
    assert first["odds_h"] is not None
    assert first["odds_over25"] is not None
    # takvim yılı sezonları (İskandinavya)
    assert first["season"] in {"2024", "2025"}


def test_parse_extra_skips_unplayed():
    csv = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Norway,Eliteserien,2025,01/04/2025,18:00,Molde,Brann,2,1,H\n"
        "Norway,Eliteserien,2025,02/04/2025,18:00,Viking,Brann,,,\n"
    )
    rows = dfetch.parse_extra_csv_bytes(csv, "NOR")
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Molde"


def test_parse_extra_wrong_format_raises():
    # 'main' biçim CSV'si extra ayrıştırıcıya verilirse net hata vermeli
    with pytest.raises(dfetch.DataFetchError):
        dfetch.parse_extra_csv_bytes(SAMPLE.read_bytes(), "NOR")


def test_parse_any_autodetects_both_formats():
    main_rows = dfetch.parse_any_csv_bytes(SAMPLE.read_bytes(), "T1")
    extra_rows = dfetch.parse_any_csv_bytes(SAMPLE_EXTRA.read_bytes(), "NOR")
    assert len(main_rows) > 100 and len(extra_rows) > 100
    assert main_rows[0]["league"] == "T1"
    assert extra_rows[0]["league"] == "NOR"


def test_parse_any_unknown_format_raises():
    with pytest.raises(dfetch.DataFetchError):
        dfetch.parse_any_csv_bytes("Foo,Bar\n1,2\n", "T1")


# --------------------------------------------------------------------------- #
# lig kayıt defteri
# --------------------------------------------------------------------------- #
def test_league_registry_covers_spor_toto_countries():
    from fpredict import config
    # Spor Toto listelerinde sık geçen ülkeler tanımlı olmalı
    for key in ["B1", "DNK", "N1", "SWE", "NOR", "POL", "P1", "T1"]:
        assert key in config.LEAGUES, key
        assert config.LEAGUES[key].source in {"main", "extra"}


def test_league_labels_unique():
    from fpredict import config
    labels = [config.league_label(k) for k in config.LEAGUES]
    assert len(labels) == len(set(labels))


# --------------------------------------------------------------------------- #
# Yedek ayna kaynağı ve ağ hatası teşhisi
# --------------------------------------------------------------------------- #
def test_mirror_url_for_covered_leagues():
    from fpredict import config
    assert config.has_mirror("E0")
    url = config.mirror_url("E0", "2324")
    assert url is not None and url.endswith("premier-league/season-2324.csv")
    # ayna yalnızca 5 büyük ligi kapsar
    assert not config.has_mirror("NOR")
    assert config.mirror_url("NOR", "2324") is None


def test_classify_network_error_messages():
    import requests

    ssl_exc = requests.exceptions.SSLError(
        "certificate verify failed: self-signed certificate"
    )
    assert "TLS" in dfetch.classify_network_error(ssl_exc)

    to_exc = requests.exceptions.ConnectTimeout("Connection timed out")
    assert "zaman aşımı" in dfetch.classify_network_error(to_exc)

    rst_exc = requests.exceptions.ConnectionError(
        "Connection aborted, ConnectionResetError(10054, ...)"
    )
    msg = dfetch.classify_network_error(rst_exc)
    assert "reset" in msg or "kapatıldı" in msg


def test_download_with_fallback_uses_second_source(monkeypatch):
    """Birinci kaynak başarısızsa ikinciye (aynaya) düşmeli."""
    calls = []

    def fake_download(url, *a, **kw):
        calls.append(url)
        if "football-data.co.uk" in url:
            raise dfetch.DataFetchError("engellendi")
        return b"x" * 100

    monkeypatch.setattr(dfetch, "download_csv", fake_download)
    content, label = dfetch.download_with_fallback([
        ("football-data.co.uk", "https://www.football-data.co.uk/x.csv"),
        ("GitHub aynası", "https://raw.githubusercontent.com/y.csv"),
    ])
    assert label == "GitHub aynası"
    assert len(calls) == 2


def test_download_with_fallback_all_fail_reports_each(monkeypatch):
    def fake_download(url, *a, **kw):
        raise dfetch.DataFetchError(f"hata: {url}")

    monkeypatch.setattr(dfetch, "download_csv", fake_download)
    with pytest.raises(dfetch.DataFetchError) as exc:
        dfetch.download_with_fallback([("A", "u1"), ("B", "u2")])
    assert "A:" in str(exc.value) and "B:" in str(exc.value)
