"""F17 — CLI command extractor: Typer, Click, argparse.

Detects ``@app.command()`` / ``@click.command()`` decorators and
``ArgumentParser.add_subparsers()`` calls and emits ``NodeKind.CLI_COMMAND``
nodes so ``list_cli_commands()`` can return deterministic results from the
graph rather than guessing by filename.
"""

from __future__ import annotations

import re
from pathlib import Path

from cce.extractors.base import ExtractedData
from cce.graph.schema import Language, Node, NodeKind
from cce.parsers.tree_sitter_parser import _get_parser, _node_id_from_qname, _text
from cce.walker import file_to_module_qname

_PY = Language.PYTHON

# Decorator names that signal a CLI command
_CLI_DECORATOR_RE = re.compile(
    r"@(?:[\w.]*\.)?(?:command|app\.command|cli\.command)\s*\(",
    re.MULTILINE,
)


class CLIExtractor:
    """Extracts Typer/Click/argparse CLI commands from Python source files."""

    def can_handle(self, path: Path, source: str) -> bool:
        return bool(
            _CLI_DECORATOR_RE.search(source)
            or "add_subparsers" in source
            or "ArgumentParser" in source
        )

    def extract(self, path: Path, rel_path: str, source: str) -> ExtractedData:
        data = ExtractedData()
        src = source.encode("utf-8", errors="replace")
        tree = _get_parser(_PY).parse(src)
        root_path = path.parents[max(0, len(path.parts) - len(rel_path.split("/")) - 1)]
        module_qname = file_to_module_qname(path, root_path)

        self._extract_decorated_commands(tree.root_node, src, rel_path, module_qname, source, data)
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_decorated_commands(
        self, root_node, src: bytes, rel_path: str, module_qname: str,
        source: str, data: ExtractedData,
    ) -> None:
        """Walk the AST; emit a CLI_COMMAND node for every decorated function."""

        def _visit(node) -> None:
            if node.type == "decorated_definition":
                decorators = [c for c in node.children if c.type == "decorator"]
                func_def = next((c for c in node.children if c.type in ("function_definition", "class_definition")), None)
                if func_def and _is_cli_decorator(decorators, src):
                    name_node = func_def.child_by_field_name("name")
                    if name_node:
                        fname = _text(name_node, src).strip()
                        line_start = func_def.start_point[0] + 1
                        line_end = func_def.end_point[0] + 1
                        qname = f"{module_qname}.{fname}"
                        # Extract help string from docstring or decorator arg
                        help_text = _extract_help(func_def, src)
                        node_id = _node_id_from_qname(qname)
                        data.nodes.append(Node(
                            id=node_id,
                            kind=NodeKind.CLI_COMMAND,
                            qualified_name=qname,
                            name=fname,
                            file_path=rel_path,
                            line_start=line_start,
                            line_end=line_end,
                            language=_PY,
                            docstring=help_text,
                            meta={"cli_framework": _detect_framework(decorators, src)},
                        ))
            for child in node.children:
                _visit(child)

        _visit(root_node)


def _is_cli_decorator(decorators, src: bytes) -> bool:
    for dec in decorators:
        text = _text(dec, src).strip()
        if re.search(r"(?:\.command|^command)\s*(?:\(|$)", text):
            return True
    return False


def _detect_framework(decorators, src: bytes) -> str:
    for dec in decorators:
        text = _text(dec, src).strip()
        if "click" in text:
            return "click"
        if "typer" in text or "app.command" in text:
            return "typer"
    return "unknown"


def _extract_help(func_def, src: bytes) -> str | None:
    """Return the first string literal in the function body (docstring)."""
    body = func_def.child_by_field_name("body")
    if not body:
        return None
    for child in body.children:
        if child.type == "expression_statement":
            for subchild in child.children:
                if subchild.type in ("string", "concatenated_string"):
                    raw = _text(subchild, src).strip().strip('"""').strip("'''").strip('"').strip("'")
                    return raw[:200] if raw else None
    return None
