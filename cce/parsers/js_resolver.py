"""Phase 4 — heuristic JS/TS/TSX reference resolution via tree-sitter AST.

No type-resolution (that requires ts-morph, a future enhancement).
Produces CALLS, RENDERS, USES_HOOK edges with confidence < 1.0.
"""

from __future__ import annotations

from pathlib import Path

from cce.graph.schema import EdgeKind, Language
from cce.parsers.base import ParsedFile, RawEdge
from cce.parsers.tree_sitter_parser import _get_parser, _node_id_from_qname, _text
from cce.walker import file_to_module_qname

_HOOK_PREFIX = "use"


def resolve_js_file(parsed: ParsedFile, root: Path) -> list[RawEdge]:
    """Return CALLS, RENDERS, USES_HOOK edges for a JS/TS/TSX file."""
    src_bytes = parsed.source.encode("utf-8", errors="replace")
    lang = parsed.language
    tree = _get_parser(lang).parse(src_bytes)
    module_qname = file_to_module_qname(parsed.path, root)
    is_tsx_jsx = lang in (Language.TSX, Language.JSX)

    # Build line → enclosing symbol qname
    line_to_sym: dict[int, str] = {}
    for sym in parsed.nodes:
        for ln in range(sym.line_start, sym.line_end + 1):
            line_to_sym[ln] = sym.qualified_name

    edges: list[RawEdge] = []

    def _caller(line: int) -> str:
        return line_to_sym.get(line, module_qname)

    def visit(node) -> None:
        t = node.type

        # Plain function calls: foo(...)
        if t == "call_expression":
            fn = node.child_by_field_name("function")
            if fn:
                name = _text(fn, src_bytes).strip()
                line = node.start_point[0] + 1
                caller_qname = _caller(line)
                src_id = _node_id_from_qname(caller_qname)

                if name.startswith(_HOOK_PREFIX) and len(name) > 3 and name[3].isupper():
                    edges.append(RawEdge(
                        src_id=src_id, dst_qualified_name=name,
                        kind=EdgeKind.USES_HOOK, file_path=parsed.rel_path,
                        line=line, confidence=0.8, resolver_method="heuristic",
                    ))
                elif "." in name:
                    edges.append(RawEdge(
                        src_id=src_id, dst_qualified_name=name,
                        kind=EdgeKind.CALLS, file_path=parsed.rel_path,
                        line=line, confidence=0.6, resolver_method="heuristic",
                    ))
                elif name[0].isupper():
                    edges.append(RawEdge(
                        src_id=src_id, dst_qualified_name=name,
                        kind=EdgeKind.CALLS, file_path=parsed.rel_path,
                        line=line, confidence=0.5, resolver_method="heuristic",
                    ))

        elif is_tsx_jsx and t in ("jsx_opening_element", "jsx_self_closing_element"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _text(name_node, src_bytes).strip()
                if name and name[0].isupper():
                    line = node.start_point[0] + 1
                    caller_qname = _caller(line)
                    edges.append(RawEdge(
                        src_id=_node_id_from_qname(caller_qname), dst_qualified_name=name,
                        kind=EdgeKind.RENDERS, file_path=parsed.rel_path,
                        line=line, confidence=0.85, resolver_method="heuristic",
                    ))

        elif t == "string" and src_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace").startswith(("'/api", '"/api')):
            line = node.start_point[0] + 1
            api_path = _text(node, src_bytes).strip("\"'")
            caller_qname = _caller(line)
            edges.append(RawEdge(
                src_id=_node_id_from_qname(caller_qname),
                dst_qualified_name=f"api:{api_path}",
                kind=EdgeKind.REFERENCES, file_path=parsed.rel_path,
                line=line, confidence=0.7, resolver_method="heuristic",
            ))

        for child in node.children:
            visit(child)

    for child in tree.root_node.children:
        visit(child)

    return edges
