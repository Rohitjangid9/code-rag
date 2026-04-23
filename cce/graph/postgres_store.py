"""F34 — PostgreSQL graph store stub.

Placeholder implementation that satisfies the ``GraphStore`` protocol.

Set ``CCE_STORE__GRAPH_BACKEND=postgres`` and provide a connection DSN via
``CCE_STORE__GRAPH_DSN`` (e.g. ``postgresql://user:pw@host/db``).

Full implementation checklist:
  - asyncpg / psycopg3 connection pool
  - nodes + edges tables with GIN index on qualified_name
  - pgvector extension for vector similarity (optional)
  - LISTEN/NOTIFY for live index invalidation
"""

from __future__ import annotations

from cce.graph.schema import Edge, EdgeKind, Node, SubGraph
from cce.logging import get_logger

log = get_logger(__name__)

_NOT_IMPL = (
    "PostgreSQL graph store is not yet fully implemented. "
    "Set CCE_STORE__GRAPH_BACKEND=sqlite to use the production backend."
)


class PostgresGraphStore:
    """Stub PostgreSQL graph store (F34)."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError(
                "CCE_STORE__GRAPH_DSN must be set when using the postgres backend. "
                "Example: postgresql://user:pw@localhost/cce"
            )
        self._dsn = dsn
        log.warning("PostgresGraphStore is a stub — %s", _NOT_IMPL)

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
