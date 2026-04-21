"""Phase 5 — Kùzu graph store (optional backend, requires `pip install kuzu`).

Usage: set CCE_PATHS__GRAPH_DB to a directory path.
Falls back to sqlite_store if kuzu is not installed.
"""

from __future__ import annotations

from pathlib import Path

from cce.graph.schema import Edge, EdgeKind, Node, SubGraph
from cce.logging import get_logger

log = get_logger(__name__)

_KUZU_NODE_DDL = """
CREATE NODE TABLE IF NOT EXISTS Symbol (
    id         STRING,
    kind       STRING,
    qualified_name STRING,
    name       STRING,
    file_path  STRING,
    line_start INT64,
    line_end   INT64,
    language   STRING,
    PRIMARY KEY (id)
);
"""

_KUZU_EDGE_DDL = """
CREATE REL TABLE IF NOT EXISTS CodeEdge (
    FROM Symbol TO Symbol,
    kind       STRING,
    file_path  STRING,
    line       INT64,
    confidence DOUBLE
);
"""


class KuzuGraphStore:
    """Implements GraphStore using Kùzu (Cypher-like graph queries)."""

    def __init__(self, db_path: Path) -> None:
        try:
            import kuzu  # noqa: PLC0415

            self._db = kuzu.Database(str(db_path))
            self._conn = kuzu.Connection(self._db)
            self._conn.execute(_KUZU_NODE_DDL)
            self._conn.execute(_KUZU_EDGE_DDL)
        except ImportError:
            log.error("kuzu not installed. Install with: pip install kuzu")
            raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_node(self, node: Node) -> None:
        self._conn.execute(
            "MERGE (s:Symbol {id: $id}) "
            "SET s.kind=$kind, s.qualified_name=$qname, s.name=$name, "
            "    s.file_path=$fp, s.line_start=$ls, s.line_end=$le, s.language=$lang",
            parameters={
                "id": node.id, "kind": node.kind.value, "qname": node.qualified_name,
                "name": node.name, "fp": node.file_path,
                "ls": node.line_start, "le": node.line_end, "lang": node.language.value,
            },
        )

    def upsert_edge(self, src_id: str, dst_id: str, kind: EdgeKind,
                    file_path: str = "", line: int = 0, confidence: float = 1.0) -> None:
        self._conn.execute(
            "MATCH (a:Symbol {id: $src}), (b:Symbol {id: $dst}) "
            "MERGE (a)-[e:CodeEdge {kind: $kind}]->(b) "
            "SET e.file_path=$fp, e.line=$ln, e.confidence=$c",
            parameters={"src": src_id, "dst": dst_id, "kind": kind.value,
                        "fp": file_path, "ln": line, "c": confidence},
        )

    def delete_for_file(self, file_path: str) -> None:
        self._conn.execute(
            "MATCH (s:Symbol {file_path: $fp})-[e:CodeEdge]-() DELETE e",
            parameters={"fp": file_path},
        )
        self._conn.execute(
            "MATCH (s:Symbol {file_path: $fp}) DELETE s",
            parameters={"fp": file_path},
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_neighborhood(self, node_id: str, depth: int = 2,
                         edge_kinds: list[EdgeKind] | None = None) -> SubGraph:
        kind_filter = ""
        params: dict = {"id": node_id, "depth": depth}
        if edge_kinds:
            kind_filter = f"WHERE e.kind IN [{', '.join(repr(k.value) for k in edge_kinds)}]"
        result = self._conn.execute(
            f"MATCH p=(s:Symbol {{id: $id}})-[e:CodeEdge*1..{depth}]-(t:Symbol) "
            f"{kind_filter} RETURN t",
            parameters=params,
        )
        nodes, ids = [], set()
        while result.has_next():
            row = result.get_next()
            n = _kuzu_row_to_node(row[0])
            if n.id not in ids:
                ids.add(n.id)
                nodes.append(n)
        return SubGraph(root_id=node_id, nodes=nodes, edges=[])

    def find_callers(self, node_id: str) -> list[Node]:
        result = self._conn.execute(
            "MATCH (a:Symbol)-[e:CodeEdge {kind: 'CALLS'}]->(b:Symbol {id: $id}) RETURN a",
            parameters={"id": node_id},
        )
        return _collect_nodes(result)

    def find_callees(self, node_id: str) -> list[Node]:
        result = self._conn.execute(
            "MATCH (a:Symbol {id: $id})-[e:CodeEdge {kind: 'CALLS'}]->(b:Symbol) RETURN b",
            parameters={"id": node_id},
        )
        return _collect_nodes(result)

    def find_references(self, node_id: str) -> list[Edge]:
        return []  # full implementation requires edge hydration; deferred

    def find_implementations(self, node_id: str) -> list[Node]:
        result = self._conn.execute(
            "MATCH (a:Symbol)-[e:CodeEdge {kind: 'INHERITS'}]->(b:Symbol {id: $id}) RETURN a",
            parameters={"id": node_id},
        )
        return _collect_nodes(result)

    def get_node(self, node_id: str) -> Node | None:
        result = self._conn.execute(
            "MATCH (s:Symbol {id: $id}) RETURN s", parameters={"id": node_id}
        )
        return _collect_nodes(result)[0] if result.has_next() else None

    def find_by_qname(self, qualified_name: str) -> Node | None:
        result = self._conn.execute(
            "MATCH (s:Symbol {qualified_name: $qn}) RETURN s LIMIT 1",
            parameters={"qn": qualified_name},
        )
        nodes = _collect_nodes(result)
        return nodes[0] if nodes else None

    def resolve_qname(self, qualified_name: str) -> str | None:
        n = self.find_by_qname(qualified_name)
        return n.id if n else None


def _kuzu_row_to_node(row_dict: dict) -> Node:
    from cce.graph.schema import Language, NodeKind  # noqa: PLC0415

    return Node(
        id=row_dict["id"],
        kind=NodeKind(row_dict["kind"]),
        qualified_name=row_dict["qualified_name"],
        name=row_dict["name"],
        file_path=row_dict["file_path"],
        line_start=row_dict["line_start"],
        line_end=row_dict["line_end"],
        language=Language(row_dict["language"]),
    )


def _collect_nodes(result) -> list[Node]:
    nodes = []
    while result.has_next():
        row = result.get_next()
        nodes.append(_kuzu_row_to_node(row[0]))
    return nodes
