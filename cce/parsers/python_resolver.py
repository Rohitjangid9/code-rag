"""Phase 4 — Jedi-based reference resolution for Python files.

Produces CALLS, REFERENCES, RETURNS_TYPE, and PARAM_TYPE edges by resolving
identifiers at call sites to their definition symbols.

F27: One ``jedi.Project`` per pipeline run is cached at module level (keyed by
root path) so Jedi avoids re-scanning the virtual-environment on every file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cce.graph.schema import EdgeKind
from cce.parsers.base import ParsedFile, RawEdge
from cce.parsers.tree_sitter_parser import _get_parser, _node_id_from_qname, _text
from cce.graph.schema import Language
from cce.walker import file_to_module_qname
from cce.logging import get_logger

log = get_logger(__name__)

# ── F27: per-root jedi.Project cache ─────────────────────────────────────────
# Maps resolved root path → jedi.Project instance so the project is created
# once per indexing run rather than once per file.
_JEDI_PROJECT_CACHE: dict[str, "Any"] = {}


def _get_jedi_project(root: Path) -> "Any":
    """Return a cached ``jedi.Project`` for *root*, creating one if needed."""
    import jedi as _jedi  # noqa: PLC0415
    key = str(root.resolve())
    if key not in _JEDI_PROJECT_CACHE:
        _JEDI_PROJECT_CACHE[key] = _jedi.Project(path=key, smart_sys_path=True)
        log.debug("Created Jedi project for %s", key)
    return _JEDI_PROJECT_CACHE[key]


def clear_jedi_project_cache() -> None:
    """Evict all cached Jedi projects (call between pipeline runs if needed)."""
    _JEDI_PROJECT_CACHE.clear()


def resolve_python_file(parsed: ParsedFile, root: Path) -> list[RawEdge]:
    """Return CALLS + REFERENCES edges for a Python file using Jedi.

    Falls back to tree-sitter heuristics if Jedi is unavailable or slow.
    F27: re-uses a module-level cached ``jedi.Project`` per root path.
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
        project = _get_jedi_project(root)  # F27: cached project
        script = jedi.Script(code=source, path=str(parsed.path), project=project)
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

    # F6: REFERENCES edges — identifiers passed as args, decorators, dict/list values, getattr
    ref_sites = _find_reference_sites(tree.root_node, src_bytes, parsed, root)
    for ref_node, src_qname, confidence in ref_sites:
        line = ref_node.start_point[0] + 1
        col = ref_node.start_point[1]
        try:
            defs = script.goto(line=line, column=col, follow_imports=True, follow_builtin_imports=False)
        except Exception:  # noqa: BLE001
            continue
        for d in defs:
            if d.module_path is None:
                continue
            dst_qname = _jedi_def_to_qname(d, root)
            if dst_qname and dst_qname != src_qname:
                edges.append(RawEdge(
                    src_id=_node_id_from_qname(src_qname),
                    dst_qualified_name=dst_qname,
                    kind=EdgeKind.REFERENCES,
                    file_path=parsed.rel_path,
                    line=line,
                    resolver_method="jedi",
                    confidence=confidence,
                ))

    # F6: getattr(module, "name") dynamic lookups (heuristic, low confidence)
    for ref_node, src_qname, name in _find_getattr_references(tree.root_node, src_bytes, parsed, root):
        line = ref_node.start_point[0] + 1
        edges.append(RawEdge(
            src_id=_node_id_from_qname(src_qname),
            dst_qualified_name=name,
            kind=EdgeKind.REFERENCES,
            file_path=parsed.rel_path,
            line=line,
            resolver_method="getattr-heuristic",
            confidence=0.4,
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


# ── F6: reference site discovery ──────────────────────────────────────────────

_BUILTINS = {
    "len", "print", "range", "enumerate", "zip", "map", "filter",
    "int", "str", "list", "dict", "set", "tuple", "bool", "float",
    "type", "isinstance", "hasattr", "getattr", "super", "open",
    "max", "min", "sum", "abs", "round", "divmod", "pow",
    "sorted", "reversed", "iter", "next", "all", "any",
    "globals", "locals", "vars", "dir", "help", "repr",
    "format", "hex", "oct", "bin", "chr", "ord", "ascii",
    "callable", "classmethod", "staticmethod", "property",
    "True", "False", "None",
}


def _find_reference_sites(root_node, src: bytes, parsed: ParsedFile, root: Path):
    """Yield (node, enclosing_qname, confidence) for identifiers used as values."""
    module_qname = file_to_module_qname(parsed.path, root)
    line_to_sym: dict[int, str] = {}
    for node in parsed.nodes:
        for ln in range(node.line_start, node.line_end + 1):
            line_to_sym[ln] = node.qualified_name

    results: list[tuple] = []

    def _enclosing_qname(line: int) -> str:
        return line_to_sym.get(line, module_qname)

    def visit(node, inside_call=False):
        # Skip the function part of calls (handled as CALLS)
        if node.type == "call":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if args:
                for child in args.children:
                    visit(child, inside_call=True)
            return

        # Decorator targets
        if node.type == "decorator":
            for child in node.children:
                if child.type in ("identifier", "attribute"):
                    line = child.start_point[0] + 1
                    results.append((child, _enclosing_qname(line), 0.7))
                    break
                if child.type == "call":
                    fn = child.child_by_field_name("function")
                    if fn and fn.type in ("identifier", "attribute"):
                        line = fn.start_point[0] + 1
                        results.append((fn, _enclosing_qname(line), 0.7))
                    break
            return

        # Dictionary / list / tuple / set values
        if node.type == "pair":
            value = node.child_by_field_name("value")
            if value and value.type in ("identifier", "attribute"):
                line = value.start_point[0] + 1
                results.append((value, _enclosing_qname(line), 0.7))
            return

        if node.type in ("list", "tuple", "set"):
            for child in node.children:
                if child.type in ("identifier", "attribute"):
                    line = child.start_point[0] + 1
                    results.append((child, _enclosing_qname(line), 0.7))
            return

        # Identifier used as an argument (but not a call's function)
        if inside_call and node.type in ("identifier", "attribute"):
            name = _text(node, src).strip()
            if name not in _BUILTINS:
                line = node.start_point[0] + 1
                results.append((node, _enclosing_qname(line), 0.7))
            return

        for child in node.children:
            visit(child, inside_call)

    visit(root_node)
    return results


def _find_getattr_references(root_node, src: bytes, parsed: ParsedFile, root: Path):
    """Yield (node, enclosing_qname, string_name) for getattr(module, 'name') calls."""
    module_qname = file_to_module_qname(parsed.path, root)
    line_to_sym: dict[int, str] = {}
    for node in parsed.nodes:
        for ln in range(node.line_start, node.line_end + 1):
            line_to_sym[ln] = node.qualified_name

    results: list[tuple] = []

    def _enclosing_qname(line: int) -> str:
        return line_to_sym.get(line, module_qname)

    def visit(node):
        if node.type == "call":
            fn = node.child_by_field_name("function")
            fn_text = _text(fn, src).strip() if fn else ""
            if fn_text == "getattr":
                args = node.child_by_field_name("arguments")
                if args:
                    arg_nodes = [c for c in args.children if c.type == "argument"]
                    if len(arg_nodes) >= 2:
                        name_arg = arg_nodes[1]
                        name_text = _text(name_arg, src).strip().strip('"').strip("'")
                        if name_text and name_text not in _BUILTINS:
                            line = name_arg.start_point[0] + 1
                            results.append((name_arg, _enclosing_qname(line), name_text))
        for child in node.children:
            visit(child)

    visit(root_node)
    return results


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
                # heuristic: include all non-builtin function calls
                if name not in {"len", "print", "range", "enumerate", "zip", "map", "filter",
                                "int", "str", "list", "dict", "set", "tuple", "bool", "float",
                                "type", "isinstance", "hasattr", "getattr", "super", "open",
                                "max", "min", "sum", "abs", "round", "divmod", "pow",
                                "sorted", "reversed", "iter", "next", "all", "any",
                                "globals", "locals", "vars", "dir", "help", "repr",
                                "format", "hex", "oct", "bin", "chr", "ord", "ascii",
                                "callable", "classmethod", "staticmethod", "property"}:
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
