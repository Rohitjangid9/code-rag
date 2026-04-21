"""Phase 2 — Layer 1 lexical search via SQLite FTS5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cce.index.db import DatabaseManager


@dataclass
class LexHit:
    path: str
    snippet: str
    rank: float


class LexicalStore:
    """Wraps the ``lex_fts`` FTS5 table for file-content search."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, rel_path: str, content: str) -> None:
        """Insert or replace the full-text content for *rel_path*."""
        conn = self._db.conn
        conn.execute("DELETE FROM lex_fts WHERE path = ?", (rel_path,))
        conn.execute("INSERT INTO lex_fts(path, content) VALUES (?, ?)", (rel_path, content))
        conn.commit()

    def delete(self, rel_path: str) -> None:
        conn = self._db.conn
        conn.execute("DELETE FROM lex_fts WHERE path = ?", (rel_path,))
        conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> list[LexHit]:
        """BM25-ranked full-text search. Returns top-*k* hits."""
        conn = self._db.conn
        rows = conn.execute(
            """
            SELECT path,
                   snippet(lex_fts, 1, '[', ']', '…', 8) AS snippet,
                   rank
            FROM lex_fts
            WHERE content MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, k),
        ).fetchall()
        return [LexHit(path=r["path"], snippet=r["snippet"], rank=r["rank"]) for r in rows]

    def search_regex(self, pattern: str, root: Path, k: int = 50) -> list[LexHit]:
        """Regex literal search via ripgrep (falls back to Python grep)."""
        try:
            return self._ripgrep(pattern, root, k)
        except (FileNotFoundError, OSError):
            return self._python_grep(pattern, root, k)

    def _ripgrep(self, pattern: str, root: Path, k: int) -> list[LexHit]:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            ["rg", "--json", "-m", "1", pattern, str(root)],
            capture_output=True, text=True, timeout=15,
        )
        hits: list[LexHit] = []
        import json  # noqa: PLC0415
        for line in result.stdout.splitlines():
            if len(hits) >= k:
                break
            try:
                obj = json.loads(line)
                if obj.get("type") == "match":
                    data = obj["data"]
                    path = str(Path(data["path"]["text"]).relative_to(root))
                    text = data["lines"]["text"].rstrip()
                    hits.append(LexHit(path=path, snippet=text, rank=0.0))
            except Exception:  # noqa: BLE001
                continue
        return hits

    def _python_grep(self, pattern: str, root: Path, k: int) -> list[LexHit]:
        import re  # noqa: PLC0415

        rx = re.compile(pattern)
        hits: list[LexHit] = []
        for py_file in root.rglob("*.py"):
            if len(hits) >= k:
                break
            try:
                for line in py_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if rx.search(line):
                        hits.append(LexHit(path=str(py_file.relative_to(root)), snippet=line, rank=0.0))
                        break
            except OSError:
                continue
        return hits
