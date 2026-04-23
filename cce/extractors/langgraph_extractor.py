"""F18 — LangGraph graph extractor.

Detects ``StateGraph().add_node("x", fn)`` and ``.add_edge("a", "b")`` calls
and emits:

* A ``REFERENCES`` edge ``builder → fn`` (confidence 0.8).
* ``meta["graph_node_name"] = "x"`` stored on the *fn* symbol so
  ``list_symbols(kind="Function") + meta.graph_node_name`` returns all
  LangGraph graph nodes deterministically.

This gives the agent a deterministic path for Q6-style questions:
  grep for ``add_node`` → resolve each → enumerate.
"""

from __future__ import annotations

import re
from pathlib import Path

from cce.extractors.base import ExtractedData
from cce.graph.schema import EdgeKind, Language, Node, NodeKind
from cce.parsers.base import RawEdge
from cce.parsers.tree_sitter_parser import _get_parser, _node_id_from_qname, _text
from cce.walker import file_to_module_qname

_PY = Language.PYTHON

# Quick heuristic to avoid parsing files with no LangGraph content
_LANGGRAPH_SIGNAL_RE = re.compile(r"StateGraph|add_node|add_edge|CompiledGraph")


class LangGraphExtractor:
    """Extracts LangGraph graph structure from Python source files."""

    def can_handle(self, path: Path, source: str) -> bool:
        return bool(_LANGGRAPH_SIGNAL_RE.search(source))

    def extract(self, path: Path, rel_path: str, source: str) -> ExtractedData:
        data = ExtractedData()
        src = source.encode("utf-8", errors="replace")
        tree = _get_parser(_PY).parse(src)
        root_path = path.parents[max(0, len(path.parts) - len(rel_path.split("/")) - 1)]
        module_qname = file_to_module_qname(path, root_path)

        # Collect builder variable names (e.g. builder = StateGraph(...))
        builder_vars = _find_builder_vars(tree.root_node, src)
        # Extract add_node / add_edge calls
        _extract_add_node(tree.root_node, src, rel_path, module_qname, builder_vars, data)
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_builder_vars(root, src: bytes) -> set[str]:
    """Return set of variable names assigned a StateGraph() instance."""
    builder_vars: set[str] = set()

    def visit(node):
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left and right and right.type == "call":
                fn = right.child_by_field_name("function")
                fn_text = _text(fn, src).strip() if fn else ""
                if "StateGraph" in fn_text or "MessageGraph" in fn_text:
                    builder_vars.add(_text(left, src).strip())
        for child in node.children:
            visit(child)

    visit(root)
    return builder_vars


def _extract_add_node(
    root, src: bytes, rel_path: str, module_qname: str,
    builder_vars: set[str], data: ExtractedData,
) -> None:
    """Emit REFERENCES edges for add_node("name", fn) calls."""

    def visit(node):
        if node.type == "call":
            fn = node.child_by_field_name("function")
            if fn and fn.type == "attribute":
                method = fn.child_by_field_name("attribute")
                obj = fn.child_by_field_name("object")
                method_name = _text(method, src).strip() if method else ""
                obj_name = _text(obj, src).strip() if obj else ""

                if method_name == "add_node" and (
                    not builder_vars or obj_name in builder_vars
                ):
                    args = node.child_by_field_name("arguments")
                    if args:
                        arg_list = [
                            c for c in args.children
                            if c.type not in (",", "(", ")")
                        ]
                        if len(arg_list) >= 2:
                            graph_name_node = arg_list[0]
                            fn_arg_node = arg_list[1]
                            graph_name = _text(graph_name_node, src).strip().strip('"').strip("'")
                            fn_ref = _text(fn_arg_node, src).strip()
                            line = node.start_point[0] + 1
                            src_qname = f"{module_qname}.__graph__"
                            dst_qname = f"{module_qname}.{fn_ref}" if "." not in fn_ref else fn_ref
                            data.raw_edges.append(RawEdge(
                                src_id=_node_id_from_qname(src_qname),
                                dst_qualified_name=dst_qname,
                                kind=EdgeKind.REFERENCES,
                                file_path=rel_path,
                                line=line,
                                resolver_method="langgraph-extractor",
                                confidence=0.8,
                            ))
                            # Store graph_node_name in a sentinel node for grep
                            data.nodes.append(Node(
                                id=_node_id_from_qname(f"{dst_qname}.__graph_node__"),
                                kind=NodeKind.FUNCTION,
                                qualified_name=dst_qname,
                                name=fn_ref,
                                file_path=rel_path,
                                line_start=line,
                                line_end=line,
                                language=_PY,
                                meta={"graph_node_name": graph_name},
                            ))

        for child in node.children:
            visit(child)

    visit(root)
