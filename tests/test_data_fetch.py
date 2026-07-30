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
