"""Phase 11 — SCIP index emitter.

Walks all indexed symbols and edges, converts them to SCIP documents/occurrences,
and returns a SCIPIndex ready for JSON serialisation.

F38: REFERENCES edges are now emitted as both a SCIP relationship (on the
     source symbol) and as ``SymbolRole.READ_ACCESS`` occurrences in the
     target file, so indexers like Sourcegraph see incoming references.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cce.graph.schema import EdgeKind, Node
from cce.scip.schema import (
    SCIPDocument,
    SCIPIndex,
    SCIPMetadata,
    SCIPOccurrence,
    SCIPPosition,
    SCIPRelationship,
    SCIPSymbolInfo,
    SCIPToolInfo,
    SymbolRole,
)
from cce.logging import get_logger

log = get_logger(__name__)

_LANG_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
}

# Edge kind → SCIP relationship field
_EDGE_TO_RELATIONSHIP = {
    EdgeKind.INHERITS: "is_implementation",
    EdgeKind.ROUTES_TO: "is_reference",
    EdgeKind.CALLS: "is_reference",
    EdgeKind.RENDERS: "is_reference",
    EdgeKind.USES_MODEL: "is_type_definition",
    EdgeKind.REFERENCES: "is_reference",  # F38: explicit REFERENCES edges
}


def _scip_symbol(node: Node) -> str:
    """Convert a CCE node to a SCIP symbol descriptor string.

    Format: scip-<language> <manager> <package> <version> <descriptors>
    """
    lang = node.language.value.lower()
    parts = node.qualified_name.split(".")
    # Use first package segment as package name
    package = parts[0] if parts else "unknown"
    descriptor = "/".join(parts[1:]) if len(parts) > 1 else node.name
    return f"scip-{lang} python {package} . {descriptor}."


class SCIPEmitter:
    """Converts the CCE symbol + graph stores into a SCIPIndex."""

    def __init__(self, sym_store, graph_store) -> None:
        self._sym = sym_store
        self._graph = graph_store

    def emit(self, root: Path) -> SCIPIndex:
        """Build and return a complete SCIPIndex for the indexed codebase."""
        metadata = SCIPMetadata(
            tool_info=SCIPToolInfo(),
            project_root=root.as_uri(),
        )
        index = SCIPIndex(metadata=metadata)

        # Group nodes by file
        all_qnames = self._sym.list_qnames()
        file_nodes: dict[str, list[Node]] = defaultdict(list)
        for qname in all_qnames:
            node = self._sym.get_by_qname(qname)
            if node:
                file_nodes[node.file_path].append(node)

        # Build one SCIPDocument per file
        for rel_path, nodes in sorted(file_nodes.items()):
            nodes.sort(key=lambda n: n.line_start)
            lang = _LANG_MAP.get(nodes[0].language.value.lower(), "unknown")
            doc = SCIPDocument(relative_path=rel_path, language=lang)

            for node in nodes:
                scip_sym = _scip_symbol(node)
                pos = SCIPPosition(
                    start_line=max(0, node.line_start - 1),
                    end_line=max(0, node.line_end - 1),
                )
                # Definition occurrence
                doc.occurrences.append(SCIPOccurrence(
                    range=pos.as_list(),
                    symbol=scip_sym,
                    symbol_roles=SymbolRole.DEFINITION,
                ))
                # Symbol info (with relationships)
                relationships = self._build_relationships(node)
                doc_lines = [node.docstring] if node.docstring else []
                if node.signature:
                    doc_lines.insert(0, f"```\n{node.signature}\n```")
                doc.symbols.append(SCIPSymbolInfo(
                    symbol=scip_sym,
                    documentation=doc_lines,
                    relationships=relationships,
                ))

                # F38: emit READ_ACCESS occurrences for all incoming REFERENCES edges
                self._emit_reference_occurrences(node, doc)

            index.documents.append(doc)
            log.debug("SCIP: emitted %s (%d symbols)", rel_path, len(nodes))

        log.info("SCIP index: %d documents, %d total symbols",
                 len(index.documents),
                 sum(len(d.symbols) for d in index.documents))
        return index

    def _emit_reference_occurrences(self, node: Node, doc: SCIPDocument) -> None:
        """Emit READ_ACCESS occurrences for all nodes that *reference* this node (F38).

        This populates the ``occurrences`` array so cross-reference viewers
        (Sourcegraph, IntelliJ SCIP plugin) can show inbound references.
        """
        scip_sym = _scip_symbol(node)
        try:
            # find_callers returns nodes whose CALLS edge points here
            callers = self._graph.find_callers(node.id)
        except Exception:  # noqa: BLE001
            callers = []
        for caller in callers:
            if not caller.line_start:
                continue
            ref_pos = SCIPPosition(start_line=max(0, caller.line_start - 1))
            doc.occurrences.append(SCIPOccurrence(
                range=ref_pos.as_list(),
                symbol=scip_sym,
                symbol_roles=SymbolRole.READ_ACCESS,
            ))

    def _build_relationships(self, node: Node) -> list[SCIPRelationship]:
        """Translate outgoing graph edges to SCIP relationships."""
        relationships: list[SCIPRelationship] = []
        try:
            edges = self._graph.find_references(node.id)
        except Exception:  # noqa: BLE001
            return relationships

        for edge in edges:
            rel_field = _EDGE_TO_RELATIONSHIP.get(edge.kind)
            if not rel_field:
                continue
            dst_node = self._graph.get_node(edge.dst_id)
            if not dst_node:
                continue
            rel = SCIPRelationship(symbol=_scip_symbol(dst_node))
            setattr(rel, rel_field, True)
            relationships.append(rel)
        return relationships
