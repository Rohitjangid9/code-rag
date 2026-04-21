"""Phase 6c — React extractor: React Router routes + CALLS_API cross-stack edges.

Detects createBrowserRouter([{path, element}]) and <Route path=... element=.../>
patterns, and axios/fetch calls to /api/* endpoints.
"""

from __future__ import annotations

import re
from pathlib import Path

from cce.extractors.base import ExtractedData
from cce.graph.schema import EdgeKind, FrameworkTag, Language, Node, NodeKind
from cce.parsers.base import RawEdge
from cce.parsers.tree_sitter_parser import (
    _child_by_field,
    _get_parser,
    _node_id_from_qname,
    _text,
)
from cce.walker import file_to_module_qname

_API_CALL_RE = re.compile(
    r"""(?:axios\.(?:get|post|put|patch|delete)|fetch)\s*\(\s*['"`](/api[^'"`]*)['"`]""",
    re.MULTILINE,
)
_ROUTE_OBJ_RE = re.compile(
    r"""path\s*:\s*['"`]([^'"`]+)['"`].*?element\s*:\s*<(\w+)""",
    re.DOTALL,
)


class ReactExtractor:
    """Extracts React Router route definitions and cross-stack API call edges."""

    def can_handle(self, path: Path, source: str) -> bool:
        return path.suffix in (".tsx", ".jsx", ".ts", ".js") and (
            "createBrowserRouter" in source
            or "<Route" in source
            or "Routes" in source
            or "/api/" in source
        )

    def extract(self, path: Path, rel_path: str, source: str) -> ExtractedData:
        data = ExtractedData()
        lang = {".tsx": Language.TSX, ".jsx": Language.JSX,
                ".ts": Language.TYPESCRIPT, ".js": Language.JAVASCRIPT}.get(path.suffix, Language.TSX)
        module_qname = file_to_module_qname(
            path,
            path.parents[len(path.parts) - len(rel_path.split("/")) - 1]
            if "/" in rel_path else path.parent,
        )

        self._extract_router_config(source, rel_path, module_qname, lang, data)
        self._extract_jsx_routes(path, source, rel_path, module_qname, lang, data)
        self._extract_api_calls(source, rel_path, module_qname, data)
        return data

    # ── createBrowserRouter([{path, element}]) ─────────────────────────────────

    def _extract_router_config(self, source: str, rel_path: str, mod: str,
                                lang: Language, data: ExtractedData) -> None:
        for m in _ROUTE_OBJ_RE.finditer(source):
            route_path = m.group(1)
            component_name = m.group(2)
            line = source[: m.start()].count("\n") + 1
            qname = f"{mod}.route.{re.sub(r'[^a-zA-Z0-9_]', '_', route_path)}"
            route_id = _node_id_from_qname(qname)
            data.nodes.append(Node(
                id=route_id,
                kind=NodeKind.ROUTE,
                qualified_name=qname,
                name=route_path,
                file_path=rel_path,
                line_start=line,
                line_end=line,
                language=lang,
                framework_tag=FrameworkTag.REACT,
                meta={"path": route_path, "component": component_name},
            ))
            data.raw_edges.append(RawEdge(
                src_id=route_id,
                dst_qualified_name=component_name,
                kind=EdgeKind.RENDERS,
                file_path=rel_path,
                line=line,
                confidence=0.9,
            ))

    # ── <Route path="..." element={<Component />} /> ──────────────────────────

    def _extract_jsx_routes(self, path: Path, source: str, rel_path: str,
                             mod: str, lang: Language, data: ExtractedData) -> None:
        src_bytes = source.encode("utf-8", errors="replace")
        try:
            tree = _get_parser(lang).parse(src_bytes)
        except Exception:  # noqa: BLE001
            return

        def visit(node) -> None:
            if node.type == "jsx_opening_element":
                name_node = _child_by_field(node, "name")
                if name_node and _text(name_node, src_bytes).strip() == "Route":
                    route_path_val, component_name = None, None
                    for attr in node.children:
                        if attr.type != "jsx_attribute":
                            continue
                        attr_name_node = attr.children[0] if attr.children else None
                        if not attr_name_node:
                            continue
                        attr_name = _text(attr_name_node, src_bytes).strip()
                        val_nodes = [c for c in attr.children if c != attr_name_node and c.type not in ("=",)]
                        if not val_nodes:
                            continue
                        val_text = _text(val_nodes[-1], src_bytes).strip().strip('"\'{}')
                        if attr_name == "path":
                            route_path_val = val_text
                        elif attr_name == "element":
                            m = re.search(r"<(\w+)", val_text)
                            if m:
                                component_name = m.group(1)
                    if route_path_val:
                        line = node.start_point[0] + 1
                        qname = f"{mod}.jsxroute.{re.sub(r'[^a-zA-Z0-9_]', '_', route_path_val)}"
                        route_id = _node_id_from_qname(qname)
                        data.nodes.append(Node(
                            id=route_id, kind=NodeKind.ROUTE,
                            qualified_name=qname, name=route_path_val,
                            file_path=rel_path, line_start=line, line_end=line,
                            language=lang, framework_tag=FrameworkTag.REACT,
                            meta={"path": route_path_val, "component": component_name},
                        ))
                        if component_name:
                            data.raw_edges.append(RawEdge(
                                src_id=route_id, dst_qualified_name=component_name,
                                kind=EdgeKind.RENDERS, file_path=rel_path, line=line, confidence=0.9,
                            ))
            for child in node.children:
                visit(child)

        visit(tree.root_node)

    # ── CALLS_API cross-stack edges ────────────────────────────────────────────

    def _extract_api_calls(self, source: str, rel_path: str, mod: str, data: ExtractedData) -> None:
        src_id = _node_id_from_qname(mod)
        for m in _API_CALL_RE.finditer(source):
            api_path = m.group(1)
            line = source[: m.start()].count("\n") + 1
            data.raw_edges.append(RawEdge(
                src_id=src_id,
                dst_qualified_name=f"api:{api_path}",
                kind=EdgeKind.REFERENCES,
                file_path=rel_path,
                line=line,
                confidence=0.85,
            ))
