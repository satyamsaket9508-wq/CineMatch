"""
CineMatch History: everything a user searches or opens is stored in SQLite.
The table is defined in history.sql; the database file is cinematch_history.db.
"""
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

def _pick_db_path():
    """Store the DB next to the app; fall back to the temp folder if that folder is read-only."""
    here = Path(__file__).parent
    try:
        probe = here / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return here / "cinematch_history.db"
    except OSError:
        return Path(tempfile.gettempdir()) / "cinematch_history.db"


DB_PATH = _pick_db_path()


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# The same schema lives in history.sql for reference, but it is embedded here so the app
# never depends on a file name (Linux hosts like Streamlit Cloud are case-sensitive).
SCHEMA = """
CREATE TABLE history (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'search',
    viewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    n_picks INTEGER,
    min_rating REAL,
    genres TEXT DEFAULT ''
);
CREATE INDEX idx_history_title ON history (title);
CREATE INDEX idx_history_time ON history (viewed_at);
"""


def init_db():
    """Create the history table from history.sql the first time. Safe to call on every run."""
    with closing(_connect()) as conn, conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'history'"
        ).fetchone()
        if not exists:
            conn.executescript(SCHEMA)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS watchlist ("
            "title TEXT PRIMARY KEY, added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
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


# ───────────────────────── Watchlist ─────────────────────────
def wl_add(title):
    with closing(_connect()) as conn, conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (title) VALUES (?)", (title,))


def wl_remove(title):
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM watchlist WHERE title = ?", (title,))


def wl_has(title):
    with closing(_connect()) as conn:
        return conn.execute("SELECT 1 FROM watchlist WHERE title = ?", (title,)).fetchone() is not None


def wl_list():
    """Saved titles, newest first."""
    with closing(_connect()) as conn:
        return [r["title"] for r in conn.execute("SELECT title FROM watchlist ORDER BY rowid DESC")]


def titles_viewed(limit=200):
    """Distinct titles from history, newest first (used for personalised picks and taste profile)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT title, COUNT(*) AS times FROM history GROUP BY title ORDER BY MAX(id) DESC LIMIT ?", (limit,)
        ).fetchall()
    return [(r["title"], r["times"]) for r in rows]