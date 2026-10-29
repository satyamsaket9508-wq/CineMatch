"""
CineMatch history database (SQLite, no extra install needed).

Stores every movie you open, plus the filter settings you used, in
`cinematch_history.db` next to app.py.

Table: history
    id            INTEGER  primary key
    title         TEXT     movie title
    viewed_at     TEXT     UTC timestamp (YYYY-MM-DD HH:MM:SS)
    n_picks       INTEGER  "How many picks" slider value
    min_rating    REAL     minimum-rating filter used
    genres        TEXT     comma-separated genre filters used (may be empty)
"""
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).parent / "cinematch_history.db"


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the table if it does not exist yet. Safe to call on every run."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                viewed_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                n_picks     INTEGER,
                min_rating  REAL,
                genres      TEXT    DEFAULT ''
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_title ON history(title)")


def add_view(title, n_picks=None, min_rating=None, genres=None):
    """Record that a movie was opened."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO history (title, n_picks, min_rating, genres) VALUES (?, ?, ?, ?)",
            (title, n_picks, min_rating, ", ".join(genres or [])),
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
            "SELECT id, title, viewed_at, n_picks, min_rating, genres FROM history ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def count():
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]


def clear():
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM history")