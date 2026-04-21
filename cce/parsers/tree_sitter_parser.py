"""Phase 3 — Tree-sitter AST parser for Python, JS, TS, TSX.

Extracts classes, functions, methods, React components, and hooks.
Uses explicit AST traversal (not queries) for cross-version stability.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from cce.graph.schema import EdgeKind, FrameworkTag, Language, Node, NodeKind
from cce.parsers.base import ParsedFile, RawEdge
from cce.walker import file_to_module_qname

# ── lazy language loaders ─────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _get_language(lang: Language):
    """Return a tree-sitter Language object, cached per language."""
    from tree_sitter import Language as TSLanguage  # noqa: PLC0415

    if lang == Language.PYTHON:
        import tree_sitter_python as m  # noqa: PLC0415
        return TSLanguage(m.language())
    if lang in (Language.JAVASCRIPT, Language.JSX):
        import tree_sitter_javascript as m  # noqa: PLC0415
        return TSLanguage(m.language())
    if lang == Language.TYPESCRIPT:
        import tree_sitter_typescript as m  # noqa: PLC0415
        return TSLanguage(m.language_typescript())
    if lang == Language.TSX:
        import tree_sitter_typescript as m  # noqa: PLC0415
        return TSLanguage(m.language_tsx())
    raise ValueError(f"Unsupported language: {lang}")


@lru_cache(maxsize=8)
def _get_parser(lang: Language):
    from tree_sitter import Parser  # noqa: PLC0415
    return Parser(_get_language(lang))


# ── helpers ───────────────────────────────────────────────────────────────────

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_by_field(node, field: str):
    return node.child_by_field_name(field)


def _first_string_in_body(body_node, src: bytes) -> str | None:
    """Extract the first string literal from a function/class body (docstring)."""
    if body_node is None:
        return None
    for child in body_node.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type in ("string", "concatenated_string"):
                    raw = _text(sub, src).strip("\"'").strip()
                    # handle triple-quoted
                    raw = raw.strip('"""').strip("'''").strip()
                    return raw[:512] if raw else None
    return None


def _node_id_from_qname(qname: str) -> str:
    """Deterministic ID: sha1 prefix of qualified_name so same symbol == same id."""
    import hashlib  # noqa: PLC0415
    return "sym_" + hashlib.sha1(qname.encode()).hexdigest()[:20]


# ── Python parser ─────────────────────────────────────────────────────────────

_PY_DECORATOR_RE = re.compile(r"@(\w[\w.]*)")


def _parse_python(parsed: ParsedFile, src: bytes, root: Path) -> None:
    module_qname = file_to_module_qname(parsed.path, root)
    tree = _get_parser(Language.PYTHON).parse(src)

    def visit(node, class_stack: list[str]) -> None:
        t = node.type

        if t in ("class_definition", "function_definition"):
            name_node = _child_by_field(node, "name")
            if not name_node:
                return
            name = _text(name_node, src)
            parents = class_stack + [name]
            qname = f"{module_qname}.{'.'.join(parents)}"

            if t == "class_definition":
                body = _child_by_field(node, "body")
                doc = _first_string_in_body(body, src)
                # base classes → INHERITS raw edges
                bases_node = _child_by_field(node, "superclasses")
                if bases_node:
                    for base in bases_node.children:
                        if base.type == "identifier":
                            parsed.raw_edges.append(RawEdge(
                                src_id=_node_id_from_qname(qname),
                                dst_qualified_name=_text(base, src),
                                kind=EdgeKind.INHERITS,
                                file_path=parsed.rel_path,
                                line=node.start_point[0] + 1,
                            ))
                kind = NodeKind.CLASS
                sig = f"class {name}"
            else:
                body = _child_by_field(node, "body")
                doc = _first_string_in_body(body, src)
                params = _child_by_field(node, "parameters")
                ret = _child_by_field(node, "return_type")
                param_str = _text(params, src) if params else "()"
                ret_str = f" -> {_text(ret, src)}" if ret else ""
                sig = f"def {name}{param_str}{ret_str}"
                kind = NodeKind.METHOD if class_stack else NodeKind.FUNCTION

            sym_id = _node_id_from_qname(qname)
            parsed.nodes.append(Node(
                id=sym_id,
                kind=kind,
                qualified_name=qname,
                name=name,
                file_path=parsed.rel_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=doc,
                language=Language.PYTHON,
                content_hash=parsed.rel_path,
            ))

            new_stack = class_stack + [name] if t == "class_definition" else class_stack
            body_node = _child_by_field(node, "body")
            if body_node:
                for child in body_node.children:
                    visit(child, new_stack)
            return  # don't re-visit children via generic path

        elif t == "decorated_definition":
            for child in node.children:
                if child.type in ("class_definition", "function_definition"):
                    visit(child, class_stack)
            return

        elif t in ("import_statement", "import_from_statement"):
            _collect_import_edges(node, parsed, src, module_qname)
            return

        for child in node.children:
            visit(child, class_stack)

    for child in tree.root_node.children:
        visit(child, [])


def _collect_import_edges(node, parsed: ParsedFile, src: bytes, module_qname: str) -> None:
    """Emit RawEdge(IMPORTS) from current module to imported names."""
    src_id = _node_id_from_qname(module_qname)
    if node.type == "import_statement":
        for child in node.children:
            if child.type in ("dotted_name", "identifier"):
                parsed.raw_edges.append(RawEdge(
                    src_id=src_id,
                    dst_qualified_name=_text(child, src),
                    kind=EdgeKind.IMPORTS,
                    file_path=parsed.rel_path,
                    line=node.start_point[0] + 1,
                    confidence=0.9,
                ))
    elif node.type == "import_from_statement":
        mod_node = _child_by_field(node, "module_name")
        if mod_node:
            mod_name = _text(mod_node, src)
            for child in node.children:
                if child.type in ("dotted_name", "identifier") and child != mod_node:
                    parsed.raw_edges.append(RawEdge(
                        src_id=src_id,
                        dst_qualified_name=f"{mod_name}.{_text(child, src)}",
                        kind=EdgeKind.IMPORTS,
                        file_path=parsed.rel_path,
                        line=node.start_point[0] + 1,
                        confidence=0.9,
                    ))


# ── JS / TS / TSX parser ──────────────────────────────────────────────────────

_PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_HOOK_RE = re.compile(r"^use[A-Z]")


def _parse_js_ts(parsed: ParsedFile, src: bytes, root: Path) -> None:
    module_qname = file_to_module_qname(parsed.path, root)
    lang = parsed.language
    tree = _get_parser(lang).parse(src)
    is_tsx_jsx = lang in (Language.TSX, Language.JSX)

    def _make_node(name: str, kind: NodeKind, ts_node, sig: str, doc: str | None,
                   fw: FrameworkTag | None = None) -> Node:
        qname = f"{module_qname}.{name}"
        return Node(
            id=_node_id_from_qname(qname),
            kind=kind,
            qualified_name=qname,
            name=name,
            file_path=parsed.rel_path,
            line_start=ts_node.start_point[0] + 1,
            line_end=ts_node.end_point[0] + 1,
            signature=sig,
            docstring=doc,
            language=lang,
            framework_tag=fw,
            content_hash=parsed.rel_path,
        )

    def _has_jsx_return(fn_node) -> bool:
        """Heuristic: function contains JSX if its text has '<' followed by identifier."""
        text = _text(fn_node, src)
        return bool(re.search(r"return\s*\(?\s*<[A-Za-z]", text))

    def visit(node) -> None:
        t = node.type

        # function declaration: function foo(...) {}
        if t == "function_declaration":
            name_n = _child_by_field(node, "name")
            if name_n:
                name = _text(name_n, src)
                kind, fw = _classify_js_fn(name, node, src, is_tsx_jsx)
                parsed.nodes.append(_make_node(name, kind, node, f"function {name}(...)", None, fw))

        # const Foo = (...) => ...   or   const Foo = function(...) {}
        elif t == "lexical_declaration":
            for decl in node.children:
                if decl.type != "variable_declarator":
                    continue
                name_n = _child_by_field(decl, "name")
                val_n = _child_by_field(decl, "value")
                if not name_n or not val_n:
                    continue
                if val_n.type not in ("arrow_function", "function"):
                    continue
                name = _text(name_n, src)
                kind, fw = _classify_js_fn(name, val_n, src, is_tsx_jsx)
                parsed.nodes.append(_make_node(name, kind, node, f"const {name} = ...", None, fw))

        # class Foo { ... }
        elif t == "class_declaration":
            name_n = _child_by_field(node, "name")
            if name_n:
                name = _text(name_n, src)
                qname = f"{module_qname}.{name}"
                sym_id = _node_id_from_qname(qname)
                parsed.nodes.append(_make_node(name, NodeKind.CLASS, node, f"class {name}", None))
                # heritage → INHERITS
                hc = _child_by_field(node, "class_heritage")
                if hc:
                    for c in hc.children:
                        if c.type == "identifier":
                            parsed.raw_edges.append(RawEdge(
                                src_id=sym_id,
                                dst_qualified_name=_text(c, src),
                                kind=EdgeKind.INHERITS,
                                file_path=parsed.rel_path,
                                line=node.start_point[0] + 1,
                            ))

        # import { Foo } from './bar'
        elif t == "import_declaration":
            src_id = _node_id_from_qname(module_qname)
            src_n = _child_by_field(node, "source")
            if src_n:
                mod = _text(src_n, src).strip("'\"")
                parsed.raw_edges.append(RawEdge(
                    src_id=src_id,
                    dst_qualified_name=mod,
                    kind=EdgeKind.IMPORTS,
                    file_path=parsed.rel_path,
                    line=node.start_point[0] + 1,
                    confidence=0.9,
                ))

        for child in node.children:
            visit(child)

    for child in tree.root_node.children:
        visit(child)


def _classify_js_fn(name: str, fn_node, src: bytes, is_tsx_jsx: bool) -> tuple[NodeKind, FrameworkTag | None]:
    if _HOOK_RE.match(name):
        return NodeKind.HOOK, FrameworkTag.REACT
    if is_tsx_jsx and _PASCAL_RE.match(name):
        text = _text(fn_node, src)
        if re.search(r"return\s*\(?\s*<[A-Za-z(/]", text):
            return NodeKind.COMPONENT, FrameworkTag.REACT
    return NodeKind.FUNCTION, None


# ── Public entry point ─────────────────────────────────────────────────────────

class TreeSitterParser:
    """Implements the Parser protocol using tree-sitter."""

    def parse(self, path: Path, rel_path: str, language: Language, source: str) -> ParsedFile:
        src = source.encode("utf-8", errors="replace")
        parsed = ParsedFile(path=path, rel_path=rel_path, language=language, source=source)
        root = path.parents[len(path.parts) - len(rel_path.split("/")) - 1] if "/" in rel_path else path.parent

        try:
            if language == Language.PYTHON:
                _parse_python(parsed, src, root)
            elif language in (Language.JAVASCRIPT, Language.JSX, Language.TYPESCRIPT, Language.TSX):
                _parse_js_ts(parsed, src, root)
        except Exception as exc:  # noqa: BLE001
            from cce.logging import get_logger  # noqa: PLC0415
            get_logger(__name__).warning("tree-sitter parse error %s: %s", rel_path, exc)

        return parsed
