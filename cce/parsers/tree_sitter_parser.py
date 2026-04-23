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
    """Return a tree-sitter Language object, cached per language.

    F29: Go, Java, and Rust grammars are loaded lazily and gracefully skipped
    when the optional tree-sitter-* packages are absent.
    """
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
    if lang == Language.GO:
        import tree_sitter_go as m  # noqa: PLC0415
        return TSLanguage(m.language())
    if lang == Language.JAVA:
        import tree_sitter_java as m  # noqa: PLC0415
        return TSLanguage(m.language())
    if lang == Language.RUST:
        import tree_sitter_rust as m  # noqa: PLC0415
        return TSLanguage(m.language())
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


_repo_salt: "ContextVar[str]" = None  # type: ignore[assignment]


def set_repo_salt(salt: str) -> "object":
    """F-M15: set the repo-scoped salt used by :func:`_node_id_from_qname`.

    Returns a token that can be passed to :func:`reset_repo_salt` to restore
    the previous value — mirroring ``contextvars.ContextVar.set`` semantics.
    Repeated calls from nested contexts stack correctly.
    """
    from contextvars import ContextVar  # noqa: PLC0415
    global _repo_salt
    if _repo_salt is None:
        _repo_salt = ContextVar("cce_repo_salt", default="")
    return _repo_salt.set(salt or "")


def reset_repo_salt(token) -> None:
    """Restore the salt to the value recorded when ``token`` was issued."""
    if _repo_salt is not None and token is not None:
        _repo_salt.reset(token)


def _current_salt() -> str:
    return _repo_salt.get() if _repo_salt is not None else ""


def _node_id_from_qname(qname: str) -> str:
    """Deterministic ID: sha1 prefix of ``<repo_salt>:<qname>`` → same symbol == same id.

    F-M15: the salt defaults to the empty string so single-repo deployments
    produce identical IDs to pre-M15 indexes.  Centralised deployments that
    index multiple repos into one database set a non-empty salt per repo via
    :func:`set_repo_salt` so qualified-name collisions don't merge symbols
    that live in different codebases.
    """
    import hashlib  # noqa: PLC0415
    salt = _current_salt()
    key = f"{salt}:{qname}" if salt else qname
    return "sym_" + hashlib.sha1(key.encode()).hexdigest()[:20]


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
                                dst_qualified_name=f"{module_qname}.{_text(base, src)}",
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
        elif t == "import_statement":
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


# ── F29: generic Go / Java / Rust parser ─────────────────────────────────────

_GO_FUNC_RE = re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", re.MULTILINE)
_JAVA_METHOD_RE = re.compile(
    r"(?:public|private|protected|static|final|abstract|synchronized|native|default)?\s+"
    r"(?:\w+(?:<[^>]*>)?\s+)(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?[{;]",
    re.MULTILINE,
)
_RUST_FN_RE = re.compile(r"^(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]", re.MULTILINE)


def _parse_generic(parsed: ParsedFile, src: bytes, root: Path, language: Language) -> None:
    """Tree-sitter based parser for Go, Java, and Rust (F29).

    Extracts top-level function/method definitions using AST node types common
    across these languages.  Raises ``ImportError`` when the grammar package is
    not installed so the caller can fall back to the heuristic regex parser.
    """
    from cce.walker import file_to_module_qname  # noqa: PLC0415

    tree = _get_parser(language).parse(src)
    module_qname = file_to_module_qname(parsed.path, root)
    source_text = src.decode("utf-8", errors="replace")

    # Node type names that represent function/method definitions per language
    fn_types: set[str] = {
        Language.GO:   {"function_declaration", "method_declaration"},
        Language.JAVA: {"method_declaration", "constructor_declaration"},
        Language.RUST: {"function_item"},
    }.get(language, {"function_declaration"})

    kind_map: dict[str, NodeKind] = {
        "function_declaration": NodeKind.FUNCTION,
        "method_declaration": NodeKind.METHOD,
        "constructor_declaration": NodeKind.METHOD,
        "function_item": NodeKind.FUNCTION,
    }

    def visit(node) -> None:
        if node.type in fn_types:
            name_node = node.child_by_field_name("name")
            if name_node:
                fname = _text(name_node, src)
                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1
                qname = f"{module_qname}.{fname}"
                parsed.nodes.append(Node(
                    id=_node_id_from_qname(qname),
                    kind=kind_map.get(node.type, NodeKind.FUNCTION),
                    qualified_name=qname,
                    name=fname,
                    file_path=parsed.rel_path,
                    line_start=line_start,
                    line_end=line_end,
                    language=language,
                ))
        for child in node.children:
            visit(child)

    visit(tree.root_node)


def _parse_heuristic(parsed: ParsedFile, source: str) -> None:
    """Regex-based fallback parser used when a tree-sitter grammar isn't installed (F29)."""
    from cce.walker import file_to_module_qname  # noqa: PLC0415

    lang = parsed.language
    module_qname = file_to_module_qname(parsed.path, parsed.path.parent)
    lines = source.splitlines()

    if lang == Language.GO:
        pattern = _GO_FUNC_RE
    elif lang == Language.JAVA:
        pattern = _JAVA_METHOD_RE
    elif lang == Language.RUST:
        pattern = _RUST_FN_RE
    else:
        return

    for m in pattern.finditer(source):
        fname = m.group(1)
        line_start = source[: m.start()].count("\n") + 1
        line_end = min(line_start + 20, len(lines))
        qname = f"{module_qname}.{fname}"
        parsed.nodes.append(Node(
            id=_node_id_from_qname(qname),
            kind=NodeKind.FUNCTION,
            qualified_name=qname,
            name=fname,
            file_path=parsed.rel_path,
            line_start=line_start,
            line_end=line_end,
            language=lang,
        ))


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
            elif language in (Language.GO, Language.JAVA, Language.RUST):
                # F29: attempt grammar-based parse; fall back to heuristic on ImportError
                try:
                    _parse_generic(parsed, src, root, language)
                except ImportError:
                    _parse_heuristic(parsed, source)
        except Exception as exc:  # noqa: BLE001
            from cce.logging import get_logger  # noqa: PLC0415
            get_logger(__name__).warning("tree-sitter parse error %s: %s", rel_path, exc)

        return parsed
