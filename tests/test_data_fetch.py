"""Veri çekme / CSV ayrıştırma testleri (ağ gerektirmez)."""
from datetime import date
from pathlib import Path

import pytest

from fpredict import data_fetch as dfetch


SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "T1_sample.csv"


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
