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

        self._extract_url_patterns(tree.root_node, src, rel_path, module_qname, data)
        self._extract_models(tree.root_node, src, rel_path, module_qname, data)
        self._extract_serializers(tree.root_node, src, rel_path, module_qname, data)
        self._extract_middleware(tree.root_node, src, rel_path, module_qname, data)
        self._extract_signals(tree.root_node, src, rel_path, module_qname, data)
        return data

    # ── URL patterns ──────────────────────────────────────────────────────────

    def _extract_url_patterns(self, root, src: bytes, rel_path: str, mod: str, data: ExtractedData) -> None:
        """Find urlpatterns = [...] and emit URLPattern nodes + ROUTES_TO edges."""
        for node in root.children:
            if node.type != "expression_statement":
                continue
            for child in node.children:
                if child.type == "assignment":
                    lhs = _child_by_field(child, "left")
                    rhs = _child_by_field(child, "right")
                    if lhs and rhs and _text(lhs, src).strip() == "urlpatterns":
                        self._parse_url_list(rhs, src, rel_path, mod, data, prefix="")

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
