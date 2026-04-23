"""F34 — DuckDB graph store stub.

Placeholder implementation that satisfies the ``GraphStore`` protocol.
Full DuckDB implementation is deferred; this stub logs a clear error when
any method is invoked so the team can track adoption.

Set ``CCE_STORE__GRAPH_BACKEND=duckdb`` to activate.
Full implementation checklist:
  - CREATE TABLE nodes / edges using DuckDB's columnar engine
  - FTS via duckdb-fts extension for qname lookups
  - ACID transactions for batch upserts
  - Persistent file at ``CCE_STORE__GRAPH_DSN`` (default: .cce/graph.duckdb)
"""

from __future__ import annotations

from pathlib import Path

from cce.graph.schema import Edge, EdgeKind, Node, SubGraph
from cce.logging import get_logger

log = get_logger(__name__)

_NOT_IMPL = (
    "DuckDB graph store is not yet fully implemented. "
    "Set CCE_STORE__GRAPH_BACKEND=sqlite to use the production backend."
)


class DuckDBGraphStore:
    """Stub DuckDB graph store (F34)."""

    def __init__(self, dsn: str = "") -> None:
        self._dsn = dsn or ".cce/graph.duckdb"
        log.warning("DuckDBGraphStore is a stub — %s", _NOT_IMPL)

    def upsert_node(self, node: Node) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def upsert_edge(self, src_id, dst_id, kind, file_path="", line=0, confidence=1.0) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def delete_for_file(self, file_path: str) -> None:
        raise NotImplementedError(_NOT_IMPL)

    def get_neighborhood(self, node_id, depth=2, edge_kinds=None) -> SubGraph:
        raise NotImplementedError(_NOT_IMPL)

    def find_callers(self, node_id: str) -> list[Node]:
        raise NotImplementedError(_NOT_IMPL)

    def find_callees(self, node_id: str) -> list[Node]:
        raise NotImplementedError(_NOT_IMPL)

    def find_references(self, node_id: str) -> list[Edge]:
        raise NotImplementedError(_NOT_IMPL)

    def find_implementations(self, node_id: str) -> list[Node]:
        raise NotImplementedError(_NOT_IMPL)

    def get_node(self, node_id: str) -> Node | None:
        raise NotImplementedError(_NOT_IMPL)

    def find_by_qname(self, qualified_name: str) -> Node | None:
        raise NotImplementedError(_NOT_IMPL)

    def resolve_qname(self, qualified_name: str) -> str | None:
        raise NotImplementedError(_NOT_IMPL)
