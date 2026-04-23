"""Phase 5 — SQLite graph store using recursive CTEs for neighborhood queries."""

from __future__ import annotations

import json

from cce.graph.base import GraphStore
from cce.graph.schema import Edge, EdgeKind, Language, Node, NodeKind, SubGraph
from cce.index.db import DatabaseManager
from cce.index.symbol_store import _row_to_node


def _synthesize_module_node(src_id: str, file_path: str) -> Node:
    """Build a placeholder Module Node for edges whose src is module-scope code."""
    norm = file_path.replace("\\", "/")
    stem = norm.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    qname = norm.rsplit(".", 1)[0].replace("/", ".") or stem
    ext = norm.rsplit(".", 1)[-1].lower() if "." in norm else ""
    lang = {
        "py": Language.PYTHON,
        "js": Language.JAVASCRIPT,
        "ts": Language.TYPESCRIPT,
        "tsx": Language.TSX,
        "jsx": Language.JSX,
    }.get(ext, Language.PYTHON)
    return Node(
        id=src_id,
        kind=NodeKind.MODULE,
        qualified_name=qname,
        name=stem,
        file_path=file_path,
        line_start=0,
        line_end=0,
        language=lang,
    )


class SQLiteGraphStore:
    """Implements GraphStore using the `edges` table and recursive CTEs."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_node(self, node: Node) -> None:
        # Nodes are owned by SymbolStore; graph store just uses their IDs.
        pass

    def upsert_edge(
        self,
        src_id: str,
        dst_id: str,
        kind: EdgeKind,
        file_path: str = "",
        line: int = 0,
        confidence: float = 1.0,
    ) -> None:
        conn = self._db.conn
        # Avoid exact duplicates — include (file_path, line) so two distinct
        # usage sites (e.g. REFERENCES at different lines) are preserved.
        exists = conn.execute(
            "SELECT 1 FROM edges WHERE src_id=? AND dst_id=? AND kind=? "
            "AND file_path=? AND line=?",
            (src_id, dst_id, kind.value, file_path, line),
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO edges (src_id, dst_id, kind, file_path, line, confidence) VALUES (?,?,?,?,?,?)",
                (src_id, dst_id, kind.value, file_path, line, confidence),
            )
            conn.commit()

    def delete_for_file(self, file_path: str) -> None:
        conn = self._db.conn
        conn.execute(
            "DELETE FROM edges WHERE file_path = ? OR "
            "src_id IN (SELECT id FROM symbols WHERE file_path=?) OR "
            "dst_id IN (SELECT id FROM symbols WHERE file_path=?)",
            (file_path, file_path, file_path),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Read — graph traversal
    # ------------------------------------------------------------------

    def get_neighborhood(
        self,
        node_id: str,
        depth: int = 2,
        edge_kinds: list[EdgeKind] | None = None,
    ) -> SubGraph:
        conn = self._db.conn

        kind_filter = ""
        params: list = [node_id, depth]
        if edge_kinds:
            placeholders = ",".join("?" * len(edge_kinds))
            kind_filter = f"AND e.kind IN ({placeholders})"
            params = [node_id, depth] + [k.value for k in edge_kinds]

        rows = conn.execute(
            f"""
            WITH RECURSIVE hood(id, depth) AS (
                SELECT ?, 0
                UNION
                SELECT e.dst_id, h.depth + 1
                FROM   edges e
                JOIN   hood h ON e.src_id = h.id
                WHERE  h.depth < ? {kind_filter}
                UNION
                SELECT e.src_id, h.depth + 1
                FROM   edges e
                JOIN   hood h ON e.dst_id = h.id
                WHERE  h.depth < ? {kind_filter}
            )
            SELECT DISTINCT s.*
            FROM   symbols s
            JOIN   hood h ON s.id = h.id
            """,
            params + (params[1:] if not edge_kinds else [depth] + [k.value for k in edge_kinds]),
        ).fetchall()

        node_ids = {r["id"] for r in rows}
        nodes = [_row_to_node(r) for r in rows]

        edge_rows = conn.execute(
            "SELECT * FROM edges WHERE src_id IN ({p}) AND dst_id IN ({p})".format(
                p=",".join("?" * len(node_ids))
            ),
            list(node_ids) * 2,
        ).fetchall() if node_ids else []

        edges = [_row_to_edge(r) for r in edge_rows]
        return SubGraph(root_id=node_id, nodes=nodes, edges=edges)

    def find_callers(self, node_id: str, include_refs: bool = True) -> list[Node]:
        kinds = [EdgeKind.CALLS.value, EdgeKind.REFERENCES.value] if include_refs else [EdgeKind.CALLS.value]
        placeholders = ",".join("?" * len(kinds))
        rows = self._db.conn.execute(
            f"SELECT e.src_id AS _edge_src_id, e.file_path AS _edge_file_path, s.* "
            f"FROM edges e LEFT JOIN symbols s ON s.id = e.src_id "
            f"WHERE e.dst_id = ? AND e.kind IN ({placeholders})",
            (node_id, *kinds),
        ).fetchall()
        out: list[Node] = []
        for r in rows:
            if r["id"] is not None:
                out.append(_row_to_node(r))
            else:
                out.append(_synthesize_module_node(r["_edge_src_id"], r["_edge_file_path"] or ""))
        return out

    def find_callees(self, node_id: str) -> list[Node]:
        rows = self._db.conn.execute(
            "SELECT s.* FROM symbols s "
            "JOIN edges e ON s.id = e.dst_id "
            "WHERE e.src_id = ? AND e.kind = ?",
            (node_id, EdgeKind.CALLS.value),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def find_references(self, node_id: str) -> list[Edge]:
        rows = self._db.conn.execute(
            "SELECT * FROM edges WHERE dst_id = ?", (node_id,)
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def find_implementations(self, node_id: str) -> list[Node]:
        rows = self._db.conn.execute(
            "SELECT s.* FROM symbols s "
            "JOIN edges e ON s.id = e.src_id "
            "WHERE e.dst_id = ? AND e.kind = ?",
            (node_id, EdgeKind.INHERITS.value),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def get_node(self, node_id: str) -> Node | None:
        row = self._db.conn.execute(
            "SELECT * FROM symbols WHERE id = ?", (node_id,)
        ).fetchone()
        return _row_to_node(row) if row else None

    def find_by_qname(self, qualified_name: str) -> Node | None:
        row = self._db.conn.execute(
            "SELECT * FROM symbols WHERE qualified_name = ? LIMIT 1", (qualified_name,)
        ).fetchone()
        return _row_to_node(row) if row else None

    def resolve_qname(self, qualified_name: str) -> str | None:
        row = self._db.conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = ? LIMIT 1", (qualified_name,)
        ).fetchone()
        return row["id"] if row else None


def _row_to_edge(row) -> Edge:
    from cce.graph.schema import Location  # noqa: PLC0415

    return Edge(
        src_id=row["src_id"],
        dst_id=row["dst_id"],
        kind=EdgeKind(row["kind"]),
        location=Location(file=row["file_path"] or "", line=row["line"] or 0, col=row["col"] or 0),
        confidence=row["confidence"],
    )
