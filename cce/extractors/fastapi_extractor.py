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
# F-M11: ``@app.websocket("/ws")`` registers a websocket handler; treat it as a
# route with method "WEBSOCKET" so downstream tooling can list it uniformly.
_WS_METHOD = "websocket"


class FastAPIExtractor:
    """Extracts FastAPI framework symbols from Python source files."""

    def can_handle(self, path: Path, source: str) -> bool:
        return (
            "FastAPI(" in source
            or "APIRouter(" in source
            or "@app." in source
            or "@router." in source
            or "BaseModel" in source
            or "from pydantic" in source
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
        # F-M11: programmatic registration and mounts
        self._extract_add_api_route(tree.root_node, src, rel_path, module_qname, router_vars, data)
        self._extract_mounts(tree.root_node, src, rel_path, module_qname, router_vars, data)
        return data

    # ── Router variable discovery ─────────────────────────────────────────────

    def _collect_router_vars(self, root, src: bytes) -> dict[str, dict]:
        """Return {var_name: {'kind': 'app'|'router', 'prefix': str}}.

        The ``prefix`` is the ``prefix=...`` kwarg passed to ``APIRouter(...)`` /
        ``FastAPI(...)``; empty string when absent. Routes declared against a
        router with a prefix get that prefix prepended to their ``meta.path``.
        """
        vars_: dict[str, dict] = {}
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
                call_text = _text(rhs, src)
                pm = re.search(r"""prefix\s*=\s*['"]([^'"]*)['"]""", call_text)
                prefix = pm.group(1) if pm else ""
                if fn_name == "FastAPI":
                    vars_[var_name] = {"kind": "app", "prefix": prefix}
                elif fn_name in ("APIRouter", "router"):
                    vars_[var_name] = {"kind": "router", "prefix": prefix}
        return vars_

    # ── Route extraction ──────────────────────────────────────────────────────

    def _extract_routes(self, root, src: bytes, rel_path: str, mod: str,
                        router_vars: dict[str, dict], data: ExtractedData) -> None:
        # Router var → include_router prefix (discovered from
        # `app.include_router(users_router, prefix="/api/v1")`).
        include_prefixes = self._collect_include_prefixes(root, src)

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
                local_path = route_meta.get("path", "/")
                router_var = route_meta.get("router_var", "")
                methods = route_meta.get("methods", [])
                response_model = route_meta.get("response_model")

                # Build the effective path: include_router.prefix + router.prefix + local.
                router_prefix = router_vars.get(router_var, {}).get("prefix", "")
                include_prefix = include_prefixes.get(router_var, "")
                full_path = _join_paths(include_prefix, router_prefix, local_path)

                qname = f"{mod}.route.{re.sub(r'[^a-zA-Z0-9_]', '_', full_path)}.{'_'.join(methods)}"
                route_id = _node_id_from_qname(qname)

                data.nodes.append(Node(
                    id=route_id,
                    kind=NodeKind.ROUTE,
                    qualified_name=qname,
                    name=full_path,
                    file_path=rel_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    language=_PY,
                    framework_tag=FrameworkTag.FASTAPI,
                    meta={
                        "path": full_path,
                        "local_path": local_path,
                        "router_var": router_var,
                        "router_module": mod,
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

    def _collect_include_prefixes(self, root, src: bytes) -> dict[str, str]:
        """Scan `app.include_router(router, prefix="/api/v1")` calls."""
        full = src.decode("utf-8", errors="replace")
        prefixes: dict[str, str] = {}
        for m in re.finditer(
            r"\w+\.include_router\(\s*(\w+)\s*(?:,\s*prefix\s*=\s*['\"]([^'\"]*)['\"])?",
            full,
        ):
            router_var, prefix = m.group(1), m.group(2) or ""
            if prefix:
                prefixes[router_var] = prefix
        return prefixes

    def _parse_route_decorator(self, dec_node, src: bytes, router_vars: dict) -> dict | None:
        """Parse @app.get('/path', response_model=X) → {path, methods, response_model, router_var}.

        F-M11: also recognises ``@app.websocket("/ws")`` — the returned
        ``methods`` list contains a single ``"WEBSOCKET"`` entry.
        """
        dec_text = _text(dec_node, src).lstrip("@").strip()
        for var in router_vars:
            for method in (*_HTTP_METHODS, _WS_METHOD):
                prefix = f"{var}.{method}"
                if dec_text.startswith(prefix):
                    m = re.search(r"""['"]([^'"]+)['"]""", dec_text)
                    path = m.group(1) if m else "/"
                    # Match an identifier optionally followed by one level of
                    # [...] — stop before the next kwarg or the closing paren.
                    rm = re.search(
                        r"response_model\s*=\s*([\w\.]+(?:\[[^\]]*\])?)", dec_text,
                    )
                    response_model = rm.group(1).strip() if rm else None
                    return {
                        "path": path,
                        "methods": [method.upper()],
                        "response_model": response_model,
                        "router_var": var,
                    }
        return None

    # ── add_api_route (programmatic registration) ─────────────────────────────

    def _extract_add_api_route(
        self, root, src: bytes, rel_path: str, mod: str,
        router_vars: dict[str, dict], data: ExtractedData,
    ) -> None:
        """F-M11: parse ``router.add_api_route("/path", handler, methods=["GET"])``.

        Emits a Route node + ROUTES_TO edge to the referenced handler symbol.
        The handler name is captured as-is; the graph resolver reconciles it to
        a qualified name during reference resolution.
        """
        full = src.decode("utf-8", errors="replace")
        include_prefixes = self._collect_include_prefixes(root, src)
        pattern = re.compile(
            r"(\w+)\.add_api_route\s*\(\s*"
            r"""['"]([^'"]+)['"]\s*,\s*"""       # path literal
            r"(\w+)"                              # endpoint identifier
            r"(?:\s*,\s*methods\s*=\s*\[([^\]]*)\])?",
        )
        for m in pattern.finditer(full):
            router_var, path, endpoint, methods_raw = (
                m.group(1), m.group(2), m.group(3), m.group(4) or "",
            )
            if router_var not in router_vars:
                continue
            methods = [
                s.strip().strip("'\"").upper()
                for s in methods_raw.split(",") if s.strip()
            ] or ["GET"]
            router_prefix = router_vars.get(router_var, {}).get("prefix", "")
            include_prefix = include_prefixes.get(router_var, "")
            full_path = _join_paths(include_prefix, router_prefix, path)
            line = full[: m.start()].count("\n") + 1
            qname = (
                f"{mod}.route.{re.sub(r'[^a-zA-Z0-9_]', '_', full_path)}."
                f"{'_'.join(methods)}"
            )
            route_id = _node_id_from_qname(qname)
            handler_qname = f"{mod}.{endpoint}"
            data.nodes.append(Node(
                id=route_id,
                kind=NodeKind.ROUTE,
                qualified_name=qname,
                name=full_path,
                file_path=rel_path,
                line_start=line,
                line_end=line,
                language=_PY,
                framework_tag=FrameworkTag.FASTAPI,
                meta={
                    "path": full_path,
                    "local_path": path,
                    "router_var": router_var,
                    "router_module": mod,
                    "methods": methods,
                    "handler": handler_qname,
                    "registration": "add_api_route",
                },
            ))
            data.raw_edges.append(RawEdge(
                src_id=route_id,
                dst_qualified_name=handler_qname,
                kind=EdgeKind.ROUTES_TO,
                file_path=rel_path,
                line=line,
            ))

    # ── Mounts (sub-apps, static files) ──────────────────────────────────────

    def _extract_mounts(
        self, root, src: bytes, rel_path: str, mod: str,
        router_vars: dict[str, dict], data: ExtractedData,
    ) -> None:
        """F-M11: parse ``app.mount("/static", StaticFiles(...))`` or sub-apps.

        Emits a MOUNTS_ROUTER edge from the parent app to the mounted object.
        The destination qualified name is ``{mod}.{target_expr}`` for local
        bindings, or the raw callable text when the target is a call expression.
        """
        full = src.decode("utf-8", errors="replace")
        pattern = re.compile(
            r"(\w+)\.mount\s*\(\s*"
            r"""['"]([^'"]+)['"]\s*,\s*"""       # mount path literal
            r"([\w\.]+)",                         # sub-app identifier / dotted name
        )
        for m in pattern.finditer(full):
            app_var, mount_path, target = m.group(1), m.group(2), m.group(3)
            if app_var not in router_vars:
                continue
            line = full[: m.start()].count("\n") + 1
            data.raw_edges.append(RawEdge(
                src_id=_node_id_from_qname(f"{mod}.{app_var}"),
                dst_qualified_name=f"{mod}.{target}",
                kind=EdgeKind.MOUNTS_ROUTER,
                file_path=rel_path,
                line=line,
                confidence=0.8,
            ))
            # Record the mount path on the router_prefixes dict so the indexer's
            # cross-file pass can stamp effective_path on any routes inside the
            # mounted sub-app.
            data.router_prefixes[f"{mod}.{target}"] = mount_path

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
            if prefix:
                data.router_prefixes[f"{mod}.{router_var}"] = prefix


def _join_paths(*parts: str) -> str:
    """Join URL-path segments, preserving an intentional trailing slash.

    ``_join_paths("/api/v1", "/users", "/")``   -> ``"/api/v1/users/"``
    ``_join_paths("/api/v1", "/users", "")``    -> ``"/api/v1/users"``
    ``_join_paths("", "/users", "{id}")``       -> ``"/users/{id}"``
    ``_join_paths("/api", "", "")``             -> ``"/api"``
    """
    cleaned = [p for p in parts if p]
    if not cleaned:
        return "/"
    last = cleaned[-1]
    segments = [seg for p in cleaned for seg in p.split("/") if seg]
    path = "/" + "/".join(segments) if segments else "/"
    if last.endswith("/") and path != "/":
        path += "/"
    return path
