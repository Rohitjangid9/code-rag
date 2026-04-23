"""Phase 4 — heuristic JS/TS/TSX reference resolution via tree-sitter AST.

No type-resolution (that requires ts-morph, a future enhancement).
Produces CALLS, RENDERS, USES_HOOK edges with confidence < 1.0.

F-M14: when an ``import`` statement binds a local name to a module-relative
path, the resolver now emits the *qualified* target (``<module_qname>.Name``)
rather than the bare identifier.  This makes JS/TS CALLS / RENDERS edges
resolvable against the symbols table just like Python CALLS.
"""

from __future__ import annotations

import posixpath
from pathlib import Path

from cce.graph.schema import EdgeKind, Language
from cce.parsers.base import ParsedFile, RawEdge
from cce.parsers.tree_sitter_parser import _get_parser, _node_id_from_qname, _text
from cce.walker import file_to_module_qname

_HOOK_PREFIX = "use"


# ── import-table construction ────────────────────────────────────────────────

def _resolve_import_target(current_module: str, spec: str) -> str:
    """Convert an import specifier into a module qualified name.

    Relative specifiers (``./foo`` / ``../bar/baz``) are resolved against the
    caller's module qname; bare/package specifiers (``react``, ``axios``) are
    returned as-is so bundler-provided packages still show up in the edge.
    """
    spec = spec.strip()
    if not spec.startswith("."):
        return spec
    # Walk up from the current module's directory component-by-component.
    base_parts = current_module.split(".")[:-1]
    cleaned = spec.lstrip("./").replace("/", ".")
    up_count = 0
    while spec.startswith("../"):
        spec = spec[3:]
        up_count += 1
    if up_count:
        base_parts = base_parts[: max(0, len(base_parts) - up_count)]
    # Drop any lingering "./" prefix component.
    tail = posixpath.normpath(cleaned).replace("/", ".").lstrip(".")
    return ".".join([*base_parts, tail]).strip(".")


def _build_import_table(tree_root, src: bytes, module_qname: str) -> dict[str, str]:
    """Return a mapping ``{local_name: qualified_target}`` for this file.

    Supported forms:
        * ``import { A, B as C } from "./mod"``   → ``A`` / ``C``
        * ``import Default from "./mod"``          → ``Default`` → ``mod.default``
        * ``import * as ns from "./mod"``          → ``ns``      → ``mod``
        * ``import "./side-effect"``               → no bindings
    """
    table: dict[str, str] = {}
    for node in tree_root.children:
        if node.type != "import_statement":
            continue
        source_node = node.child_by_field_name("source")
        if not source_node:
            continue
        raw_spec = _text(source_node, src).strip().strip("'\"`")
        target_module = _resolve_import_target(module_qname, raw_spec)

        for child in node.children:
            if child.type == "import_clause":
                for sub in child.children:
                    if sub.type == "identifier":
                        # Default import: ``import Foo from "..."``.  The local
                        # name is bound to the module itself (rather than
                        # ``<module>.default``) so that
                        # ``import axios from "axios"; axios.get(...)`` resolves
                        # to ``axios.get`` — matching how the ecosystem uses
                        # default exports as the module's public surface.
                        table[_text(sub, src).strip()] = target_module
                    elif sub.type == "namespace_import":
                        # ``* as ns``: the only identifier child is the alias
                        for n in sub.children:
                            if n.type == "identifier":
                                table[_text(n, src).strip()] = target_module
                    elif sub.type == "named_imports":
                        for spec in sub.children:
                            if spec.type != "import_specifier":
                                continue
                            name_node = spec.child_by_field_name("name")
                            alias_node = spec.child_by_field_name("alias")
                            if not name_node:
                                continue
                            imported = _text(name_node, src).strip()
                            local = _text(alias_node, src).strip() if alias_node else imported
                            table[local] = f"{target_module}.{imported}"
    return table


def _qualify(name: str, import_table: dict[str, str]) -> tuple[str, bool]:
    """Rewrite ``name`` (or ``ns.member``) using the import table if possible.

    Returns ``(qualified_name, resolved)`` — ``resolved`` is ``True`` when the
    head of ``name`` was found in the import table, regardless of whether the
    substitution changed the text (e.g. ``axios`` imported from ``"axios"``
    produces the same string, but the resolution is still considered deliberate
    and gets the higher-confidence ``resolver_method="import"`` tag).
    """
    if name in import_table:
        return import_table[name], True
    if "." in name:
        head, _, tail = name.partition(".")
        if head in import_table:
            return f"{import_table[head]}.{tail}", True
    return name, False


# ── main entry point ─────────────────────────────────────────────────────────

def resolve_js_file(parsed: ParsedFile, root: Path) -> list[RawEdge]:
    """Return CALLS, RENDERS, USES_HOOK, REFERENCES edges for a JS/TS/TSX file."""
    src_bytes = parsed.source.encode("utf-8", errors="replace")
    lang = parsed.language
    tree = _get_parser(lang).parse(src_bytes)
    module_qname = file_to_module_qname(parsed.path, root)
    is_tsx_jsx = lang in (Language.TSX, Language.JSX)

    import_table = _build_import_table(tree.root_node, src_bytes, module_qname)

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
                qualified, was_resolved = _qualify(name, import_table)
                resolved = "import" if was_resolved else "heuristic"

                if name.startswith(_HOOK_PREFIX) and len(name) > 3 and name[3].isupper():
                    edges.append(RawEdge(
                        src_id=src_id, dst_qualified_name=qualified,
                        kind=EdgeKind.USES_HOOK, file_path=parsed.rel_path,
                        line=line,
                        confidence=0.9 if resolved == "import" else 0.8,
                        resolver_method=resolved,
                    ))
                elif "." in name:
                    edges.append(RawEdge(
                        src_id=src_id, dst_qualified_name=qualified,
                        kind=EdgeKind.CALLS, file_path=parsed.rel_path,
                        line=line,
                        confidence=0.8 if resolved == "import" else 0.6,
                        resolver_method=resolved,
                    ))
                elif name[0].isupper():
                    edges.append(RawEdge(
                        src_id=src_id, dst_qualified_name=qualified,
                        kind=EdgeKind.CALLS, file_path=parsed.rel_path,
                        line=line,
                        confidence=0.8 if resolved == "import" else 0.5,
                        resolver_method=resolved,
                    ))

        elif is_tsx_jsx and t in ("jsx_opening_element", "jsx_self_closing_element"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _text(name_node, src_bytes).strip()
                if name and name[0].isupper():
                    line = node.start_point[0] + 1
                    caller_qname = _caller(line)
                    qualified, was_resolved = _qualify(name, import_table)
                    resolved = "import" if was_resolved else "heuristic"
                    edges.append(RawEdge(
                        src_id=_node_id_from_qname(caller_qname), dst_qualified_name=qualified,
                        kind=EdgeKind.RENDERS, file_path=parsed.rel_path,
                        line=line,
                        confidence=0.95 if resolved == "import" else 0.85,
                        resolver_method=resolved,
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
