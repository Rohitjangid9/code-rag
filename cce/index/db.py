"""SQLite connection manager with schema migrations for all layers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- Phase 1: file tracking
CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    mtime       REAL NOT NULL,
    lang        TEXT NOT NULL,
    size        INTEGER DEFAULT 0,
    indexed_at  REAL NOT NULL
);

-- Phase 2: lexical full-text search (porter tokeniser)
CREATE VIRTUAL TABLE IF NOT EXISTS lex_fts USING fts5(
    path     UNINDEXED,
    content,
    tokenize='porter unicode61'
);

-- Phase 3: symbol index
CREATE TABLE IF NOT EXISTS symbols (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    name          TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    line_start    INTEGER NOT NULL,
    line_end      INTEGER NOT NULL,
    signature     TEXT,
    docstring     TEXT,
    language      TEXT NOT NULL,
    framework_tag TEXT,
    visibility    TEXT DEFAULT 'public',
    content_hash  TEXT DEFAULT '',
    meta          TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sym_qname ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_sym_file  ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_sym_name  ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_sym_kind  ON symbols(kind);

-- FTS5 over symbol fields
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    id             UNINDEXED,
    name,
    qualified_name,
    docstring,
    signature,
    content        = 'symbols',
    content_rowid  = 'rowid',
    tokenize       = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS sym_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, id, name, qualified_name, docstring, signature)
    VALUES (new.rowid, new.id, new.name, new.qualified_name, new.docstring, new.signature);
END;

CREATE TRIGGER IF NOT EXISTS sym_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, id, name, qualified_name, docstring, signature)
    VALUES ('delete', old.rowid, old.id, old.name, old.qualified_name, old.docstring, old.signature);
END;

CREATE TRIGGER IF NOT EXISTS sym_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, id, name, qualified_name, docstring, signature)
    VALUES ('delete', old.rowid, old.id, old.name, old.qualified_name, old.docstring, old.signature);
    INSERT INTO symbols_fts(rowid, id, name, qualified_name, docstring, signature)
    VALUES (new.rowid, new.id, new.name, new.qualified_name, new.docstring, new.signature);
END;

-- Phase 4 / 5: code graph edges
CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id     TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    file_path  TEXT,
    line       INTEGER,
    col        INTEGER DEFAULT 0,
    confidence REAL    DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_edge_src  ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edge_dst  ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edge_kind ON edges(kind);
CREATE INDEX IF NOT EXISTS idx_edge_srcdk ON edges(src_id, kind);
CREATE INDEX IF NOT EXISTS idx_edge_dstdk ON edges(dst_id, kind);

-- Schema version for incremental rebuild detection
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
INSERT OR IGNORE INTO meta VALUES ('schema_version', '1');
"""


class DatabaseManager:
    """Manages a single SQLite connection with lazy initialisation."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._run_schema()
        return self._conn

    def _run_schema(self) -> None:
        for stmt in _SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)  # type: ignore[union-attr]
        self._conn.commit()  # type: ignore[union-attr]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


_managers: dict[Path, DatabaseManager] = {}


def get_db(db_path: Path) -> DatabaseManager:
    """Return a cached DatabaseManager for the given path."""
    if db_path not in _managers:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _managers[db_path] = DatabaseManager(db_path)
    return _managers[db_path]
