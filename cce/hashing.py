"""File content hashing and change-detection against the SQLite files table."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from cce.index.db import DatabaseManager
from cce.walker import WalkedFile


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ChangeSet:
    new: list[WalkedFile] = field(default_factory=list)
    changed: list[WalkedFile] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)   # repo-relative paths

    @property
    def to_index(self) -> list[WalkedFile]:
        return self.new + self.changed


def detect_changes(files: list[WalkedFile], db: DatabaseManager, root: Path) -> ChangeSet:
    """Compare *files* against the stored hashes; return what must be re-indexed."""
    conn = db.conn

    # Load all stored paths → hash
    stored: dict[str, str] = {
        row["path"]: row["content_hash"]
        for row in conn.execute("SELECT path, content_hash FROM files")
    }

    cs = ChangeSet()
    seen: set[str] = set()

    for wf in files:
        rel = str(wf.rel_path)
        seen.add(rel)
        current_hash = sha256_file(wf.path)
        if rel not in stored:
            cs.new.append(wf)
            _upsert_file(wf, current_hash, db, root)
        elif stored[rel] != current_hash:
            cs.changed.append(wf)
            _upsert_file(wf, current_hash, db, root)

    cs.deleted = [p for p in stored if p not in seen]
    return cs


def _upsert_file(wf: WalkedFile, content_hash: str, db: DatabaseManager, root: Path) -> None:
    from ulid import ULID  # noqa: PLC0415
    conn = db.conn
    rel = str(wf.rel_path)
    stat = wf.path.stat()
    conn.execute(
        """
        INSERT INTO files (id, path, content_hash, mtime, lang, size, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            content_hash = excluded.content_hash,
            mtime        = excluded.mtime,
            size         = excluded.size,
            indexed_at   = excluded.indexed_at
        """,
        (str(ULID()), rel, content_hash, stat.st_mtime, wf.language.value, stat.st_size, time.time()),
    )
    conn.commit()


def delete_file_records(rel_path: str, db: DatabaseManager) -> None:
    """Remove all index data owned by *rel_path* (symbols, edges, lex, files)."""
    conn = db.conn
    # Remove symbols owned by this file (triggers cascade to symbols_fts)
    conn.execute("DELETE FROM symbols WHERE file_path = ?", (rel_path,))
    # Remove edges referencing those symbols
    conn.execute(
        "DELETE FROM edges WHERE src_id NOT IN (SELECT id FROM symbols) "
        "   OR dst_id NOT IN (SELECT id FROM symbols)"
    )
    # Remove lexical entry
    conn.execute("DELETE FROM lex_fts WHERE path = ?", (rel_path,))
    # Remove file record
    conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))
    conn.commit()
