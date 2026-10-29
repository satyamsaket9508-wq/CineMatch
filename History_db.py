"""
CineMatch History: everything a user searches or opens is stored in SQLite.
The table is defined in history.sql; the database file is cinematch_history.db.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).parent / "cinematch_history.db"


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


SCHEMA_FILE = Path(__file__).parent / "history.sql"


def init_db():
    """Create the history table from history.sql the first time. Safe to call on every run."""
    with closing(_connect()) as conn, conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'history'"
        ).fetchone()
        if not exists:
            conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        # upgrade an older database that was created before the `source` column existed
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(history)")]
        if "source" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN source TEXT NOT NULL DEFAULT 'search'")


def add_view(title, source="search", n_picks=None, min_rating=None, genres=None):
    """Record that a movie was opened."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO history (title, source, n_picks, min_rating, genres) VALUES (?, ?, ?, ?, ?)",
            (title, source, n_picks, min_rating, ", ".join(genres or [])),
        )


def recent(limit=10, unique=True):
    """Most recent views, newest first. unique=True shows each movie once."""
    with closing(_connect()) as conn:
        if unique:
            rows = conn.execute(
                """
                SELECT title, MAX(viewed_at) AS viewed_at, COUNT(*) AS times
                FROM history GROUP BY title
                ORDER BY MAX(id) DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT title, viewed_at, 1 AS times FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def most_viewed(limit=5):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT title, COUNT(*) AS times FROM history GROUP BY title "
            "ORDER BY times DESC, MAX(id) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def all_rows():
    """Full history, newest first (used for CSV download)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, title, source, viewed_at, n_picks, min_rating, genres FROM history ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def count():
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]


def clear():
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM history")