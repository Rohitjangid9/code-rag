"""Phase 6a — Django framework extractor.

Extracts: URLPattern nodes + ROUTES_TO edges, Model nodes + fields,
DRF Serializer nodes + USES_MODEL edges, Middleware nodes, HANDLES_SIGNAL edges.
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


class DjangoExtractor:
    """Extracts Django + DRF framework symbols from Python source files."""

    def can_handle(self, path: Path, source: str) -> bool:
        return (
            "urlpatterns" in source
            or "models.Model" in source
            or "ModelSerializer" in source
            or "serializers.Serializer" in source
            or ("MIDDLEWARE" in source and path.name == "settings.py")
            or "@receiver" in source
            or "admin.site.register" in source
        )

    def extract(self, path: Path, rel_path: str, source: str) -> ExtractedData:
        data = ExtractedData()
        src = source.encode("utf-8", errors="replace")
        tree = _get_parser(_PY).parse(src)
        module_qname = file_to_module_qname(path, path.parents[len(path.parts) - len(rel_path.split("/")) - 1]
                                            if "/" in rel_path else path.parent)

        # F-M12: discover DRF router prefixes *before* URL walking so nested
        # ``path("api/v1/", include(router.urls))`` patterns can attach their
        # prefix to every ``router.register(...)`` route in the same module.
        include_prefixes = self._collect_include_prefixes(tree.root_node, src)
        self._extract_url_patterns(tree.root_node, src, rel_path, module_qname, data, include_prefixes)
        self._extract_drf_router_register(tree.root_node, src, rel_path, module_qname, data, include_prefixes)
        self._extract_viewset_actions(tree.root_node, src, rel_path, module_qname, data)
        self._extract_models(tree.root_node, src, rel_path, module_qname, data)
        self._extract_serializers(tree.root_node, src, rel_path, module_qname, data)
        self._extract_middleware(tree.root_node, src, rel_path, module_qname, data)
        self._extract_signals(tree.root_node, src, rel_path, module_qname, data)
        return data

    # ── include() prefix discovery ────────────────────────────────────────────

    def _collect_include_prefixes(self, root, src: bytes) -> dict[str, str]:
        """Scan ``path("prefix/", include(<target>))`` calls.

        Returns a mapping ``{target_text: prefix}`` where ``target_text`` is
        either a router var name (``router`` when the include is
        ``include(router.urls)``), a dotted module path (``"myapp.urls"``), or
        a bare variable name.  The mapping is used both for in-file DRF router
        prefix resolution and surfaced via ``data.router_prefixes`` so the
        indexer can stamp effective paths on routes declared in other files.
        """
        full = src.decode("utf-8", errors="replace")
        prefixes: dict[str, str] = {}
        # path("prefix/", include(<target>))  — target may be router.urls, "x.urls", identifier
        pattern = re.compile(
            r"""(?:path|re_path|url)\s*\(\s*"""
            r"""r?['"]([^'"]*)['"]\s*,\s*include\s*\(\s*"""
            r"""(?:r?['"]([^'"]+)['"]|([\w\.]+))"""
        )
        for m in pattern.finditer(full):
            prefix, mod_literal, ident = m.group(1), m.group(2), m.group(3)
            if mod_literal:
                prefixes[mod_literal] = prefix
            elif ident:
                # include(router.urls) → key on the bare router var name.
                key = ident.split(".", 1)[0] if ident.endswith(".urls") else ident
                prefixes[key] = prefix
        return prefixes

    # ── URL patterns ──────────────────────────────────────────────────────────

    def _extract_url_patterns(
        self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData,
        include_prefixes: dict[str, str] | None = None,
    ) -> None:
        """Find urlpatterns = [...] and emit URLPattern nodes + ROUTES_TO edges.

        F-M12: ``path("prefix/", include("other.urls"))`` entries are recorded
        in ``data.router_prefixes`` keyed on the include target so the indexer
        can stamp ``effective_path`` on URL patterns declared in the included
        module.
        """
        include_prefixes = include_prefixes or {}
        for node in root.children:
            if node.type != "expression_statement":
                continue
            for child in node.children:
                if child.type == "assignment":
                    lhs = _child_by_field(child, "left")
                    rhs = _child_by_field(child, "right")
                    if lhs and rhs and _text(lhs, src).strip() == "urlpatterns":
                        self._parse_url_list(rhs, src, rel_path, mod, data, prefix="")
        # Surface cross-file prefixes so routes in included modules get stamped
        # with their effective path by the indexer's post-pass.
        for target, prefix in include_prefixes.items():
            if "." in target or target.endswith("urls"):
                data.router_prefixes[target] = prefix

    def _parse_url_list(self, list_node, src: bytes, rel_path: str, mod: str,
                        data: ExtractedData, prefix: str) -> None:
        if list_node.type not in ("list", "tuple"):
            return
        for item in list_node.children:
            if item.type != "call":
                continue
            fn = _child_by_field(item, "function")
            args = _child_by_field(item, "arguments")
            if not fn or not args:
                continue
            fn_name = _text(fn, src).strip()
            if fn_name not in ("path", "re_path", "url"):
                continue

            arg_children = [c for c in args.children if c.type not in (",", "(", ")")]
            if len(arg_children) < 2:
                continue

            url_pattern = _text(arg_children[0], src).strip("\"'")
            effective_path = (prefix + url_pattern).replace("//", "/")

            view_node = arg_children[1]
            view_ref = _text(view_node, src).strip()

            # Skip include(...) entries — they are resolved via router_prefixes.
            if view_ref.startswith("include("):
                continue

            # Emit URLPattern node
            qname = f"{mod}.url.{re.sub(r'[^a-zA-Z0-9_]', '_', effective_path)}"
            node_id = _node_id_from_qname(qname)
            data.nodes.append(Node(
                id=node_id,
                kind=NodeKind.URL_PATTERN,
                qualified_name=qname,
                name=effective_path,
                file_path=rel_path,
                line_start=item.start_point[0] + 1,
                line_end=item.end_point[0] + 1,
                language=_PY,
                framework_tag=FrameworkTag.DJANGO,
                meta={"pattern": effective_path, "view_ref": view_ref},
            ))
            # ROUTES_TO edge (resolved later against the view symbol)
            data.raw_edges.append(RawEdge(
                src_id=node_id,
                dst_qualified_name=view_ref.replace(".as_view()", "").strip(),
                kind=EdgeKind.ROUTES_TO,
                file_path=rel_path,
                line=item.start_point[0] + 1,
            ))

    # ── DRF router.register ──────────────────────────────────────────────────

    # Default ViewSet action → HTTP method mapping.  Action names become URL
    # suffixes except for the root list/create endpoint, which lives at "".
    _VIEWSET_ACTIONS = (
        ("list", "GET", ""),
        ("create", "POST", ""),
        ("retrieve", "GET", "{pk}/"),
        ("update", "PUT", "{pk}/"),
        ("partial_update", "PATCH", "{pk}/"),
        ("destroy", "DELETE", "{pk}/"),
    )

    def _extract_drf_router_register(
        self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData,
        include_prefixes: dict[str, str],
    ) -> None:
        """F-M12: ``router.register("prefix", ViewSet)`` → URLPattern per action.

        Emits one URLPattern + ROUTES_TO edge for each standard ViewSet action
        (list/create/retrieve/update/partial_update/destroy).  When the router
        is mounted under ``path("api/v1/", include(router.urls))`` in the same
        module, the include prefix is prepended so ``effective_path`` is
        populated without needing the cross-file stamping pass.
        """
        full = src.decode("utf-8", errors="replace")
        pattern = re.compile(
            r"(\w+)\.register\s*\(\s*"
            r"""r?['"]([^'"]+)['"]\s*,\s*"""
            r"(\w+)",
        )
        for m in pattern.finditer(full):
            router_var, url_prefix, viewset = m.group(1), m.group(2), m.group(3)
            line = full[: m.start()].count("\n") + 1
            mount_prefix = include_prefixes.get(router_var, "")
            handler_qname = f"{mod}.{viewset}"
            for action_name, http_method, suffix in self._VIEWSET_ACTIONS:
                effective_path = _join_url(mount_prefix, url_prefix, suffix)
                qname = (
                    f"{mod}.url.{re.sub(r'[^a-zA-Z0-9_]', '_', effective_path)}"
                    f".{http_method}.{action_name}"
                )
                node_id = _node_id_from_qname(qname)
                data.nodes.append(Node(
                    id=node_id,
                    kind=NodeKind.URL_PATTERN,
                    qualified_name=qname,
                    name=effective_path,
                    file_path=rel_path,
                    line_start=line,
                    line_end=line,
                    language=_PY,
                    framework_tag=FrameworkTag.DRF,
                    meta={
                        "pattern": effective_path,
                        "view_ref": viewset,
                        "viewset": handler_qname,
                        "action": action_name,
                        "methods": [http_method],
                        "router_var": router_var,
                    },
                ))
                data.raw_edges.append(RawEdge(
                    src_id=node_id,
                    dst_qualified_name=f"{handler_qname}.{action_name}",
                    kind=EdgeKind.ROUTES_TO,
                    file_path=rel_path,
                    line=line,
                    confidence=0.8,
                ))

    # ── ViewSet @action ──────────────────────────────────────────────────────

    def _extract_viewset_actions(
        self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData,
    ) -> None:
        """F-M12: emit URLPattern nodes for ``@action`` methods on ViewSets.

        ``@action(detail=True, methods=["post"])`` produces one extra endpoint
        per ViewSet method.  The effective URL is
        ``{router_prefix}/{pk}/{method_name}/`` (detail=True) or
        ``{router_prefix}/{method_name}/`` (detail=False).  We do not attempt
        to resolve the router prefix here — the indexer's cross-file stamper
        handles that using the ViewSet qualified name recorded in ``meta``.
        """
        for cls in root.children:
            if cls.type != "class_definition":
                continue
            bases = _child_by_field(cls, "superclasses")
            if not bases or "ViewSet" not in _text(bases, src):
                continue
            cls_name_node = _child_by_field(cls, "name")
            if not cls_name_node:
                continue
            cls_name = _text(cls_name_node, src)
            body = _child_by_field(cls, "body")
            if not body:
                continue
            for member in body.children:
                if member.type != "decorated_definition":
                    continue
                action_meta = self._parse_action_decorator(member, src)
                if not action_meta:
                    continue
                for fn_node in member.children:
                    if fn_node.type not in ("function_definition", "async_function_definition"):
                        continue
                    fn_name_node = _child_by_field(fn_node, "name")
                    if not fn_name_node:
                        continue
                    fn_name = _text(fn_name_node, src)
                    detail = action_meta["detail"]
                    methods = action_meta["methods"]
                    suffix = f"{{pk}}/{fn_name}/" if detail else f"{fn_name}/"
                    qname = (
                        f"{mod}.{cls_name}.action.{fn_name}."
                        f"{'_'.join(methods)}"
                    )
                    node_id = _node_id_from_qname(qname)
                    line = fn_node.start_point[0] + 1
                    data.nodes.append(Node(
                        id=node_id,
                        kind=NodeKind.URL_PATTERN,
                        qualified_name=qname,
                        name=suffix,
                        file_path=rel_path,
                        line_start=line,
                        line_end=fn_node.end_point[0] + 1,
                        language=_PY,
                        framework_tag=FrameworkTag.DRF,
                        meta={
                            "pattern": suffix,
                            "view_ref": f"{cls_name}.{fn_name}",
                            "viewset": f"{mod}.{cls_name}",
                            "action": fn_name,
                            "detail": detail,
                            "methods": methods,
                        },
                    ))
                    data.raw_edges.append(RawEdge(
                        src_id=node_id,
                        dst_qualified_name=f"{mod}.{cls_name}.{fn_name}",
                        kind=EdgeKind.ROUTES_TO,
                        file_path=rel_path,
                        line=line,
                    ))

    def _parse_action_decorator(self, decorated_node, src: bytes) -> dict | None:
        """Return ``{'detail': bool, 'methods': [str, ...]}`` for ``@action(...)``.

        Returns ``None`` when no ``@action`` decorator is present.  The
        ``methods`` list falls back to ``["GET"]`` when the kwarg is omitted
        (matching DRF's default).
        """
        for dec in decorated_node.children:
            if dec.type != "decorator":
                continue
            dec_text = _text(dec, src)
            if not re.search(r"@\s*action\b", dec_text):
                continue
            detail = bool(re.search(r"detail\s*=\s*True", dec_text))
            methods_match = re.search(r"methods\s*=\s*\[([^\]]*)\]", dec_text)
            methods = ["GET"]
            if methods_match:
                methods = [
                    s.strip().strip("'\"").upper()
                    for s in methods_match.group(1).split(",") if s.strip()
                ] or ["GET"]
            return {"detail": detail, "methods": methods}
        return None

    # ── Models ────────────────────────────────────────────────────────────────

    def _extract_models(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        for node in root.children:
            target = node
            if node.type == "decorated_definition":
                for c in node.children:
                    if c.type == "class_definition":
                        target = c
                        break
            if target.type != "class_definition":
                continue
            bases = _child_by_field(target, "superclasses")
            if not bases:
                continue
            base_text = _text(bases, src)
            if "models.Model" not in base_text and "Model" not in base_text:
                continue
            name_node = _child_by_field(target, "name")
            if not name_node:
                continue
            name = _text(name_node, src)
            qname = f"{mod}.{name}"
            fields = self._extract_model_fields(target, src)
            data.nodes.append(Node(
                id=_node_id_from_qname(qname),
                kind=NodeKind.MODEL,
                qualified_name=qname,
                name=name,
                file_path=rel_path,
                line_start=target.start_point[0] + 1,
                line_end=target.end_point[0] + 1,
                language=_PY,
                framework_tag=FrameworkTag.DJANGO,
                meta={"fields": fields},
            ))

    def _extract_model_fields(self, class_node, src: bytes) -> list[dict]:
        fields = []
        body = _child_by_field(class_node, "body")
        if not body:
            return fields
        for stmt in body.children:
            if stmt.type != "expression_statement":
                continue
            for child in stmt.children:
                if child.type == "assignment":
                    lhs = _child_by_field(child, "left")
                    rhs = _child_by_field(child, "right")
                    if lhs and rhs and rhs.type == "call":
                        field_name = _text(lhs, src).strip()
                        fn = _child_by_field(rhs, "function")
                        field_type = _text(fn, src).strip() if fn else "Field"
                        if field_name and not field_name.startswith("_"):
                            fields.append({"name": field_name, "type": field_type})
        return fields

    # ── DRF Serializers ───────────────────────────────────────────────────────

    def _extract_serializers(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        for node in root.children:
            if node.type != "class_definition":
                continue
            bases = _child_by_field(node, "superclasses")
            if not bases:
                continue
            base_text = _text(bases, src)
            if "Serializer" not in base_text:
                continue
            name_node = _child_by_field(node, "name")
            if not name_node:
                continue
            name = _text(name_node, src)
            qname = f"{mod}.{name}"
            sym_id = _node_id_from_qname(qname)
            model_name = self._find_meta_model(node, src)
            data.nodes.append(Node(
                id=sym_id,
                kind=NodeKind.SERIALIZER,
                qualified_name=qname,
                name=name,
                file_path=rel_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                language=_PY,
                framework_tag=FrameworkTag.DRF,
                meta={"model": model_name},
            ))
            if model_name:
                data.raw_edges.append(RawEdge(
                    src_id=sym_id,
                    dst_qualified_name=model_name,
                    kind=EdgeKind.USES_MODEL,
                    file_path=rel_path,
                    line=node.start_point[0] + 1,
                ))

    def _find_meta_model(self, class_node, src: bytes) -> str | None:
        body = _child_by_field(class_node, "body")
        if not body:
            return None
        for child in body.children:
            if child.type == "class_definition":
                cname = _child_by_field(child, "name")
                if cname and _text(cname, src) == "Meta":
                    meta_body = _child_by_field(child, "body")
                    if meta_body:
                        for stmt in meta_body.children:
                            for sub in stmt.children:
                                if sub.type == "assignment":
                                    lhs = _child_by_field(sub, "left")
                                    rhs = _child_by_field(sub, "right")
                                    if lhs and rhs and _text(lhs, src).strip() == "model":
                                        return _text(rhs, src).strip()
        return None

    # ── Middleware ────────────────────────────────────────────────────────────

    def _extract_middleware(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        for node in root.children:
            if node.type != "expression_statement":
                continue
            for child in node.children:
                if child.type != "assignment":
                    continue
                lhs = _child_by_field(child, "left")
                rhs = _child_by_field(child, "right")
                if not lhs or not rhs:
                    continue
                if _text(lhs, src).strip() != "MIDDLEWARE":
                    continue
                order = 0
                for item in rhs.children:
                    if item.type == "string":
                        mw_name = _text(item, src).strip("\"'")
                        qname = f"{mod}.middleware.{order}"
                        data.nodes.append(Node(
                            id=_node_id_from_qname(qname),
                            kind=NodeKind.MIDDLEWARE,
                            qualified_name=qname,
                            name=mw_name,
                            file_path=rel_path,
                            line_start=item.start_point[0] + 1,
                            line_end=item.end_point[0] + 1,
                            language=_PY,
                            framework_tag=FrameworkTag.DJANGO,
                            meta={"order": order, "dotted_path": mw_name},
                        ))
                        order += 1

    # ── Signals ───────────────────────────────────────────────────────────────

    def _extract_signals(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        """Find @receiver(signal, sender=Model) decorated functions."""
        for node in root.children:
            if node.type != "decorated_definition":
                continue
            sender = None
            for dec in node.children:
                if dec.type == "decorator":
                    dec_text = _text(dec, src)
                    if "@receiver" in dec_text:
                        m = re.search(r"sender\s*=\s*(\w+)", dec_text)
                        if m:
                            sender = m.group(1)
            for child in node.children:
                if child.type == "function_definition":
                    fn_name_node = _child_by_field(child, "name")
                    if fn_name_node:
                        fn_name = _text(fn_name_node, src)
                        fn_qname = f"{mod}.{fn_name}"
                        if sender:
                            data.raw_edges.append(RawEdge(
                                src_id=_node_id_from_qname(sender),
                                dst_qualified_name=fn_qname,
                                kind=EdgeKind.HANDLES_SIGNAL,
                                file_path=rel_path,
                                line=child.start_point[0] + 1,
                            ))


def _join_url(*parts: str) -> str:
    """Join Django URL segments preserving the trailing slash of the last part."""
    cleaned = [p for p in parts if p]
    if not cleaned:
        return ""
    trailing = cleaned[-1].endswith("/")
    segments: list[str] = []
    for p in cleaned:
        for seg in p.split("/"):
            if seg:
                segments.append(seg)
    joined = "/".join(segments)
    if trailing and joined:
        joined += "/"
    return joined
