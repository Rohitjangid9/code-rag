"""Phase 2 — Layer 1 lexical search via SQLite FTS5.

F26: lex_sym_fts provides per-symbol / 50-line-window indexing alongside the
file-level lex_fts table.  Both tables coexist; search() queries lex_fts for
broad file-level BM25 matching while search_symbols() queries lex_sym_fts for
high-precision symbol-scoped matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cce.index.db import DatabaseManager


@dataclass
class LexHit:
    path: str
    snippet: str
    rank: float
    qualified_name: str = ""
    line_start: int = 0
    line_end: int = 0


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
        # F-WIN: delete both slash variants so stale back-slash records
        # are removed when the indexer switches to POSIX rel-paths.
        win = rel_path.replace("/", "\\")
        conn.execute("DELETE FROM lex_fts WHERE path IN (?, ?)", (rel_path, win))
        conn.execute("INSERT INTO lex_fts(path, content) VALUES (?, ?)", (rel_path, content))
        conn.commit()

    def upsert_symbol(
        self,
        rel_path: str,
        qualified_name: str,
        line_start: int,
        line_end: int,
        content: str,
    ) -> None:
        """Upsert one FTS5 row for a single symbol body (F26).

        Also emits one row per 50-line window so that large symbol bodies are
        chunked into discoverable windows rather than one giant entry.
        """
        conn = self._db.conn
        # Remove old rows for this qualified_name
        conn.execute(
            "DELETE FROM lex_sym_fts WHERE qualified_name = ?", (qualified_name,)
        )
        lines = content.splitlines()
        window = 50
        if len(lines) <= window:
            conn.execute(
                "INSERT INTO lex_sym_fts(path, qualified_name, line_start, line_end, content)"
                " VALUES (?, ?, ?, ?, ?)",
                (rel_path, qualified_name, line_start, line_end, content),
            )
        else:
            for start_idx in range(0, len(lines), window):
                chunk_lines = lines[start_idx: start_idx + window]
                chunk_content = "\n".join(chunk_lines)
                chunk_line_start = line_start + start_idx
                chunk_line_end = min(line_start + start_idx + window - 1, line_end)
                conn.execute(
                    "INSERT INTO lex_sym_fts(path, qualified_name, line_start, line_end, content)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (rel_path, qualified_name, chunk_line_start, chunk_line_end, chunk_content),
                )
        conn.commit()

    def delete(self, rel_path: str) -> None:
        conn = self._db.conn
        win = rel_path.replace("/", "\\")
        conn.execute("DELETE FROM lex_fts WHERE path IN (?, ?)", (rel_path, win))
        conn.execute("DELETE FROM lex_sym_fts WHERE path IN (?, ?)", (rel_path, win))
        conn.commit()

    def search_symbols(self, query: str, k: int = 20) -> list[LexHit]:
        """BM25-ranked search over the per-symbol lex_sym_fts table (F26)."""
        tokens = re.findall(r"[\w*]+", query)
        tokens = [t for t in tokens if len(t) > 1 or "*" in t]
        if not tokens:
            return []
        safe = " OR ".join(tokens) if len(tokens) > 1 else tokens[0]
        conn = self._db.conn
        try:
            rows = conn.execute(
                """
                SELECT path, qualified_name,
                       CAST(line_start AS INTEGER) AS line_start,
                       CAST(line_end AS INTEGER) AS line_end,
                       snippet(lex_sym_fts, 4, '[', ']', '…', 8) AS snippet,
                       rank
                FROM lex_sym_fts
                WHERE content MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe, k),
            ).fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [
            LexHit(
                path=r["path"],
                snippet=r["snippet"],
                rank=r["rank"],
                qualified_name=r["qualified_name"],
                line_start=r["line_start"],
                line_end=r["line_end"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> list[LexHit]:
        """BM25-ranked full-text search. Returns top-*k* hits."""
        # Sanitize for FTS5: drop characters that confuse the query parser
        # while keeping word chars and * for prefix matching.
        # For multi-token queries we join with OR so any matching file is
        # returned; single-token queries stay as-is ( FTS5 will tokenise
        # on underscores / dots etc. via unicode61).
        tokens = re.findall(r"[\w*]+", query)
        tokens = [t for t in tokens if len(t) > 1 or "*" in t]
        if not tokens:
            return []
        if len(tokens) > 1:
            safe = " OR ".join(tokens)
        else:
            safe = tokens[0]
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
            (safe, k),
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
