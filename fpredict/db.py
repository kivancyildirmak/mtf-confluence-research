"""SQLite tabanlı yerel cache.

Her açılışta baştan indirme yapmamak için indirilen maçlar burada saklanır.
`matches` tablosu (kaynak, sezon, tarih, takımlar, skorlar, oranlar) tutar;
`meta` tablosu son güncelleme zaman damgalarını tutar.

Maçlar `match_uid` (lig|tarih|ev|deplasman) ile teklenir; böylece aynı maç
tekrar indirildiğinde çift kayıt oluşmaz (INSERT OR REPLACE).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_uid   TEXT PRIMARY KEY,
    league      TEXT NOT NULL,
    season      TEXT,
    date        TEXT NOT NULL,          -- ISO 'YYYY-MM-DD'
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_goals  INTEGER,
    away_goals  INTEGER,
    result      TEXT,                   -- H/D/A
    odds_h      REAL,
    odds_d      REAL,
    odds_a      REAL,
    odds_over25 REAL,
    odds_under25 REAL
);
CREATE INDEX IF NOT EXISTS idx_matches_league_date ON matches(league, date);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_COLUMNS = [
    "match_uid", "league", "season", "date", "home_team", "away_team",
    "home_goals", "away_goals", "result", "odds_h", "odds_d", "odds_a",
    "odds_over25", "odds_under25",
]


def make_uid(league: str, date: str, home: str, away: str) -> str:
    """Bir maç için tekil kimlik üretir."""
    return f"{league}|{date}|{home}|{away}".lower()


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Şemanın var olduğundan emin olan bir bağlantı context manager'ı."""
    config.ensure_app_dir()
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_matches(rows: Iterable[dict], db_path: Path | None = None) -> int:
    """Maç kayıtlarını ekler/günceller. Eklenen/güncellenen satır sayısını döndürür."""
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ",".join(["?"] * len(_COLUMNS))
    sql = f"INSERT OR REPLACE INTO matches ({','.join(_COLUMNS)}) VALUES ({placeholders})"
    with connect(db_path) as conn:
        conn.executemany(sql, [[r.get(c) for c in _COLUMNS] for r in rows])
    return len(rows)


def load_matches(league: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Cache'deki maçları DataFrame olarak yükler (opsiyonel lig filtresi)."""
    with connect(db_path) as conn:
        if league:
            df = pd.read_sql_query(
                "SELECT * FROM matches WHERE league = ? ORDER BY date",
                conn, params=(league,),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM matches ORDER BY date", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def set_meta(key: str, value: str, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )


def get_meta(key: str, db_path: Path | None = None) -> str | None:
    with connect(db_path) as conn:
        cur = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
    return row["value"] if row else None


def league_summary(db_path: Path | None = None) -> pd.DataFrame:
    """Her lig için: maç sayısı, tarih aralığı, son güncelleme."""
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT league,
                   COUNT(*)         AS matches,
                   MIN(date)        AS first_date,
                   MAX(date)        AS last_date
            FROM matches GROUP BY league ORDER BY league
            """,
            conn,
        )
        meta = pd.read_sql_query("SELECT key, value FROM meta", conn)
    updated = {
        k.replace("updated_", ""): v
        for k, v in zip(meta.get("key", []), meta.get("value", []))
        if isinstance(k, str) and k.startswith("updated_")
    }
    if not df.empty:
        df["last_updated"] = df["league"].map(updated).fillna("-")
    return df


def touch_updated(league: str, db_path: Path | None = None) -> None:
    set_meta(f"updated_{league}", datetime.now(timezone.utc).isoformat(timespec="seconds"), db_path)
