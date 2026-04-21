"""Phase 6b — FastAPI framework extractor.

Extracts: Route nodes + ROUTES_TO edges, PydanticModel nodes,
DEPENDS_ON edges, MOUNTS_ROUTER edges.
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

_PY = Language.PYTHON
_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


class FastAPIExtractor:
    """Extracts FastAPI framework symbols from Python source files."""

    def can_handle(self, path: Path, source: str) -> bool:
        return (
            "FastAPI" in source
            or "APIRouter" in source
            or "@app." in source
            or "@router." in source
        )

    def extract(self, path: Path, rel_path: str, source: str) -> ExtractedData:
        data = ExtractedData()
        src = source.encode("utf-8", errors="replace")
        tree = _get_parser(_PY).parse(src)
        module_qname = file_to_module_qname(
            path,
            path.parents[len(path.parts) - len(rel_path.split("/")) - 1]
            if "/" in rel_path else path.parent,
        )

        # Collect router variable names (e.g. app = FastAPI(), users_router = APIRouter())
        router_vars = self._collect_router_vars(tree.root_node, src)
        self._extract_routes(tree.root_node, src, rel_path, module_qname, router_vars, data)
        self._extract_pydantic_models(tree.root_node, src, rel_path, module_qname, data)
        self._extract_include_router(tree.root_node, src, rel_path, module_qname, data)
        return data

    # ── Router variable discovery ─────────────────────────────────────────────

    def _collect_router_vars(self, root, src: bytes) -> dict[str, str]:
        """Return {var_name: 'app'|'router'} for FastAPI()/APIRouter() assignments."""
        vars_: dict[str, str] = {}
        for node in root.children:
            if node.type != "expression_statement":
                continue
            for child in node.children:
                if child.type != "assignment":
                    continue
                lhs = _child_by_field(child, "left")
                rhs = _child_by_field(child, "right")
                if not lhs or not rhs or rhs.type != "call":
                    continue
                fn = _child_by_field(rhs, "function")
                if not fn:
                    continue
                fn_name = _text(fn, src).strip()
                var_name = _text(lhs, src).strip()
                if fn_name == "FastAPI":
                    vars_[var_name] = "app"
                elif fn_name in ("APIRouter", "router"):
                    vars_[var_name] = "router"
        return vars_

    # ── Route extraction ──────────────────────────────────────────────────────

    def _extract_routes(self, root, src: bytes, rel_path: str, mod: str,
                        router_vars: dict[str, str], data: ExtractedData) -> None:
        for node in root.children:
            if node.type != "decorated_definition":
                continue
            route_meta: dict | None = None
            for dec in node.children:
                if dec.type != "decorator":
                    continue
                meta = self._parse_route_decorator(dec, src, router_vars)
                if meta:
                    route_meta = meta
                    break
            if not route_meta:
                continue

            for child in node.children:
                if child.type not in ("function_definition", "async_function_definition"):
                    continue
                fn_name_node = _child_by_field(child, "name")
                if not fn_name_node:
                    continue
                fn_name = _text(fn_name_node, src)
                fn_qname = f"{mod}.{fn_name}"
                fn_id = _node_id_from_qname(fn_qname)
                route_path = route_meta.get("path", "/")
                methods = route_meta.get("methods", [])
                response_model = route_meta.get("response_model")

                qname = f"{mod}.route.{re.sub(r'[^a-zA-Z0-9_]', '_', route_path)}.{'_'.join(methods)}"
                route_id = _node_id_from_qname(qname)

                data.nodes.append(Node(
                    id=route_id,
                    kind=NodeKind.ROUTE,
                    qualified_name=qname,
                    name=route_path,
                    file_path=rel_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    language=_PY,
                    framework_tag=FrameworkTag.FASTAPI,
                    meta={
                        "path": route_path,
                        "methods": methods,
                        "response_model": response_model,
                        "handler": fn_qname,
                    },
                ))
                data.raw_edges.append(RawEdge(
                    src_id=route_id,
                    dst_qualified_name=fn_qname,
                    kind=EdgeKind.ROUTES_TO,
                    file_path=rel_path,
                    line=child.start_point[0] + 1,
                ))
                # Extract Depends() from parameters
                self._extract_depends(child, src, fn_id, rel_path, data)

    def _parse_route_decorator(self, dec_node, src: bytes, router_vars: dict) -> dict | None:
        """Parse @app.get('/path', response_model=X) → {path, methods, response_model}."""
        dec_text = _text(dec_node, src).lstrip("@").strip()
        for var, _ in router_vars.items():
            for method in _HTTP_METHODS:
                prefix = f"{var}.{method}"
                if dec_text.startswith(prefix):
                    m = re.search(r"""['"]([^'"]+)['"]""", dec_text)
                    path = m.group(1) if m else "/"
                    rm = re.search(r"response_model\s*=\s*(\w[\w\[\], ]*)", dec_text)
                    response_model = rm.group(1).strip() if rm else None
                    return {"path": path, "methods": [method.upper()], "response_model": response_model}
        return None

    # ── Depends extraction ────────────────────────────────────────────────────

    def _extract_depends(self, fn_node, src: bytes, fn_id: str, rel_path: str, data: ExtractedData) -> None:
        params = _child_by_field(fn_node, "parameters")
        if not params:
            return
        for param in params.children:
            param_text = _text(param, src)
            m = re.search(r"Depends\(\s*(\w+)\s*\)", param_text)
            if m:
                dep_name = m.group(1)
                data.raw_edges.append(RawEdge(
                    src_id=fn_id,
                    dst_qualified_name=dep_name,
                    kind=EdgeKind.DEPENDS_ON,
                    file_path=rel_path,
                    line=param.start_point[0] + 1,
                    confidence=0.9,
                ))

    # ── Pydantic models ───────────────────────────────────────────────────────

    def _extract_pydantic_models(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        for node in root.children:
            if node.type != "class_definition":
                continue
            bases = _child_by_field(node, "superclasses")
            if not bases or "BaseModel" not in _text(bases, src):
                continue
            name_node = _child_by_field(node, "name")
            if not name_node:
                continue
            name = _text(name_node, src)
            qname = f"{mod}.{name}"
            data.nodes.append(Node(
                id=_node_id_from_qname(qname),
                kind=NodeKind.PYDANTIC_MODEL,
                qualified_name=qname,
                name=name,
                file_path=rel_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=_PY,
                framework_tag=FrameworkTag.FASTAPI,
            ))

    # ── include_router ────────────────────────────────────────────────────────

    def _extract_include_router(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        full = src.decode("utf-8", errors="replace")
        for m in re.finditer(r"(\w+)\.include_router\(\s*(\w+)\s*(?:,\s*prefix\s*=\s*['\"]([^'\"]*)['\"])?", full):
            app_var, router_var, prefix = m.group(1), m.group(2), m.group(3) or ""
            line = full[: m.start()].count("\n") + 1
            data.raw_edges.append(RawEdge(
                src_id=_node_id_from_qname(f"{mod}.{app_var}"),
                dst_qualified_name=f"{mod}.{router_var}",
                kind=EdgeKind.MOUNTS_ROUTER,
                file_path=rel_path,
                line=line,
                confidence=0.9,
            ))
