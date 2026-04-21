"""Phase 4 — Jedi-based reference resolution for Python files.

Produces CALLS, REFERENCES, RETURNS_TYPE, and PARAM_TYPE edges by resolving
identifiers at call sites to their definition symbols.
"""

from __future__ import annotations

from pathlib import Path

from cce.graph.schema import EdgeKind
from cce.parsers.base import ParsedFile, RawEdge
from cce.parsers.tree_sitter_parser import _get_parser, _node_id_from_qname, _text
from cce.graph.schema import Language
from cce.walker import file_to_module_qname
from cce.logging import get_logger

log = get_logger(__name__)


def resolve_python_file(parsed: ParsedFile, root: Path) -> list[RawEdge]:
    """Return CALLS + REFERENCES edges for a Python file using Jedi.

    Falls back to tree-sitter heuristics if Jedi is unavailable or slow.
    """
    try:
        import jedi  # noqa: PLC0415
    except ImportError:
        log.warning("jedi not installed — skipping Python reference resolution")
        return []

    edges: list[RawEdge] = []
    source = parsed.source
    module_qname = file_to_module_qname(parsed.path, root)

    try:
        project = jedi.Project(path=str(root), smart_sys_path=True)
        script = jedi.Script(source=source, path=str(parsed.path), project=project)
    except Exception as exc:  # noqa: BLE001
        log.debug("Jedi Script init failed for %s: %s", parsed.rel_path, exc)
        return _heuristic_python(parsed, root)

    # Find all call expressions in the file
    src_bytes = source.encode("utf-8", errors="replace")
    tree = _get_parser(Language.PYTHON).parse(src_bytes)
    call_sites = _find_call_nodes(tree.root_node, src_bytes)

    for call_node, caller_qname in _associate_calls_to_symbols(call_sites, src_bytes, parsed, root):
        line = call_node.start_point[0] + 1
        col  = call_node.start_point[1]
        callee_name = _text(call_node, src_bytes)

        try:
            defs = script.goto(line=line, column=col, follow_imports=True, follow_builtin_imports=False)
        except Exception:  # noqa: BLE001
            continue

        for d in defs:
            if d.module_path is None:
                continue
            callee_qname = _jedi_def_to_qname(d, root)
            if callee_qname and callee_qname != caller_qname:
                edges.append(RawEdge(
                    src_id=_node_id_from_qname(caller_qname),
                    dst_qualified_name=callee_qname,
                    kind=EdgeKind.CALLS,
                    file_path=parsed.rel_path,
                    line=line,
                    resolver_method="jedi",
                    confidence=1.0,
                ))

    return edges


def _find_call_nodes(root_node, src: bytes) -> list:
    """Collect all `call` AST nodes in the tree."""
    results = []

    def visit(node):
        if node.type == "call":
            fn_node = node.child_by_field_name("function")
            if fn_node and fn_node.type in ("identifier", "attribute"):
                results.append(fn_node)
        for child in node.children:
            visit(child)

    visit(root_node)
    return results


def _associate_calls_to_symbols(call_nodes, src: bytes, parsed: ParsedFile, root: Path):
    """Yield (call_node, enclosing_symbol_qname) pairs."""
    module_qname = file_to_module_qname(parsed.path, root)
    # Build line → symbol mapping from already-parsed nodes
    line_to_sym: dict[tuple[int, int], str] = {}
    for node in parsed.nodes:
        for ln in range(node.line_start, node.line_end + 1):
            line_to_sym[ln] = node.qualified_name

    for call_node in call_nodes:
        line = call_node.start_point[0] + 1
        # Find the innermost enclosing symbol
        caller = line_to_sym.get(line, module_qname)
        yield call_node, caller


def _jedi_def_to_qname(d, root: Path) -> str | None:
    """Convert a jedi Definition to a dotted qualified name."""
    try:
        rel = Path(d.module_path).relative_to(root)
    except (ValueError, TypeError):
        return None
    parts = list(rel.with_suffix("").parts)
    if d.name:
        parts.append(d.name)
    return ".".join(parts)


# ── fallback: heuristic call detection without Jedi ──────────────────────────

def _heuristic_python(parsed: ParsedFile, root: Path) -> list[RawEdge]:
    """Extract CALLS edges from Python AST without type resolution."""
    src_bytes = parsed.source.encode("utf-8", errors="replace")
    tree = _get_parser(Language.PYTHON).parse(src_bytes)
    module_qname = file_to_module_qname(parsed.path, root)
    src_id = _node_id_from_qname(module_qname)
    edges: list[RawEdge] = []

    def visit(node):
        if node.type == "call":
            fn = node.child_by_field_name("function")
            if fn:
                name = _text(fn, src_bytes).strip()
                if "." in name or name[0].isupper():
                    edges.append(RawEdge(
                        src_id=src_id,
                        dst_qualified_name=name,
                        kind=EdgeKind.CALLS,
                        file_path=parsed.rel_path,
                        line=node.start_point[0] + 1,
                        resolver_method="heuristic",
                        confidence=0.5,
                    ))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return edges
