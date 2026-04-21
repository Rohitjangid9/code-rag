"""GraphStore protocol — implemented by sqlite_store.py and kuzu_store.py."""

from __future__ import annotations

from typing import Protocol

from cce.graph.schema import Edge, EdgeKind, Node, SubGraph


class GraphStore(Protocol):
    """Minimal graph interface shared by all backends."""

    def upsert_node(self, node: Node) -> None: ...

    def upsert_edge(
        self,
        src_id: str,
        dst_id: str,
        kind: EdgeKind,
        file_path: str = "",
        line: int = 0,
        confidence: float = 1.0,
    ) -> None: ...

    def delete_for_file(self, file_path: str) -> None:
        """Remove all nodes AND all incident edges for a given file."""
        ...

    def get_neighborhood(
        self,
        node_id: str,
        depth: int = 2,
        edge_kinds: list[EdgeKind] | None = None,
    ) -> SubGraph: ...

    def find_callers(self, node_id: str) -> list[Node]: ...

    def find_callees(self, node_id: str) -> list[Node]: ...

    def find_references(self, node_id: str) -> list[Edge]: ...

    def find_implementations(self, node_id: str) -> list[Node]: ...

    def get_node(self, node_id: str) -> Node | None: ...

    def find_by_qname(self, qualified_name: str) -> Node | None: ...

    def resolve_qname(self, qualified_name: str) -> str | None:
        """Return the node id for *qualified_name*, or None."""
        ...
