-- CineMatch history database (SQLite)
-- Every movie a user searches for or opens is stored in the history table.
-- source is one of: search, click, surprise

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