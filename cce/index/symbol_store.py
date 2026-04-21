"""Phase 3 — Layer 2 symbol CRUD + FTS5 search."""

from __future__ import annotations

import json
from dataclasses import dataclass

from cce.graph.schema import Node
from cce.index.db import DatabaseManager


@dataclass
class SymbolHit:
    node: Node
    rank: float
    match_field: str  # which field matched (name / qualified_name / docstring)


class SymbolStore:
    """Wraps the ``symbols`` table and its FTS5 companion."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, node: Node) -> None:
        conn = self._db.conn
        conn.execute(
            """
            INSERT INTO symbols
                (id, kind, qualified_name, name, file_path, line_start, line_end,
                 signature, docstring, language, framework_tag, visibility, content_hash, meta)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                kind          = excluded.kind,
                qualified_name= excluded.qualified_name,
                name          = excluded.name,
                file_path     = excluded.file_path,
                line_start    = excluded.line_start,
                line_end      = excluded.line_end,
                signature     = excluded.signature,
                docstring     = excluded.docstring,
                language      = excluded.language,
                framework_tag = excluded.framework_tag,
                visibility    = excluded.visibility,
                content_hash  = excluded.content_hash,
                meta          = excluded.meta
            """,
            (
                node.id, node.kind.value, node.qualified_name, node.name,
                node.file_path, node.line_start, node.line_end,
                node.signature, node.docstring, node.language.value,
                node.framework_tag.value if node.framework_tag else None,
                node.visibility, node.content_hash, json.dumps(node.meta),
            ),
        )
        conn.commit()

    def upsert_many(self, nodes: list[Node]) -> None:
        conn = self._db.conn
        conn.executemany(
            """
            INSERT INTO symbols
                (id, kind, qualified_name, name, file_path, line_start, line_end,
                 signature, docstring, language, framework_tag, visibility, content_hash, meta)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind, qualified_name=excluded.qualified_name,
                name=excluded.name, file_path=excluded.file_path,
                line_start=excluded.line_start, line_end=excluded.line_end,
                signature=excluded.signature, docstring=excluded.docstring,
                language=excluded.language, framework_tag=excluded.framework_tag,
                visibility=excluded.visibility, content_hash=excluded.content_hash,
                meta=excluded.meta
            """,
            [
                (
                    n.id, n.kind.value, n.qualified_name, n.name,
                    n.file_path, n.line_start, n.line_end,
                    n.signature, n.docstring, n.language.value,
                    n.framework_tag.value if n.framework_tag else None,
                    n.visibility, n.content_hash, json.dumps(n.meta),
                )
                for n in nodes
            ],
        )
        conn.commit()

    def delete_for_file(self, file_path: str) -> None:
        conn = self._db.conn
        conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_qname(self, qualified_name: str) -> Node | None:
        row = self._db.conn.execute(
            "SELECT * FROM symbols WHERE qualified_name = ? LIMIT 1", (qualified_name,)
        ).fetchone()
        return _row_to_node(row) if row else None

    def get_by_id(self, node_id: str) -> Node | None:
        row = self._db.conn.execute(
            "SELECT * FROM symbols WHERE id = ? LIMIT 1", (node_id,)
        ).fetchone()
        return _row_to_node(row) if row else None

    def get_for_file(self, file_path: str) -> list[Node]:
        rows = self._db.conn.execute(
            "SELECT * FROM symbols WHERE file_path = ? ORDER BY line_start", (file_path,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def search(self, query: str, k: int = 20) -> list[SymbolHit]:
        rows = self._db.conn.execute(
            """
            SELECT s.*, f.rank
            FROM symbols_fts f
            JOIN symbols s ON s.rowid = f.rowid
            WHERE symbols_fts MATCH ?
            ORDER BY f.rank
            LIMIT ?
            """,
            (query, k),
        ).fetchall()
        return [SymbolHit(node=_row_to_node(r), rank=r["rank"], match_field="fts") for r in rows]

    def list_qnames(self) -> list[str]:
        return [r[0] for r in self._db.conn.execute("SELECT qualified_name FROM symbols")]


def _row_to_node(row) -> Node:
    from cce.graph.schema import FrameworkTag, Language, NodeKind  # noqa: PLC0415

    return Node(
        id=row["id"],
        kind=NodeKind(row["kind"]),
        qualified_name=row["qualified_name"],
        name=row["name"],
        file_path=row["file_path"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        signature=row["signature"],
        docstring=row["docstring"],
        language=Language(row["language"]),
        framework_tag=FrameworkTag(row["framework_tag"]) if row["framework_tag"] else None,
        visibility=row["visibility"],
        content_hash=row["content_hash"] or "",
        meta=json.loads(row["meta"] or "{}"),
    )
