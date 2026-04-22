"""Typed tool surface used by CLI, HTTP API, and LangGraph agents.

Phases 1-5 are implemented. Phase 6+ functions (get_route, get_component_tree,
get_api_flow, semantic search) raise NotImplementedError until those phases land.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from cce.graph.schema import Edge, EdgeKind, Location, Node, SubGraph


# ── Response models (stable public API) ───────────────────────────────────────

class Hit(BaseModel):
    node: Node | None = None
    path: str
    line_start: int
    line_end: int
    snippet: str
    score: float
    provenance: Literal["lex", "vec", "graph", "hybrid"] = "hybrid"


class RouteInfo(BaseModel):
    pattern: str
    methods: list[str] = Field(default_factory=list)
    handler_qname: str
    framework: str
    request_model: str | None = None
    response_model: str | None = None


class ComponentTree(BaseModel):
    component_qname: str
    children: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)


class CrossStackFlow(BaseModel):
    anchor: str
    steps: list[dict] = Field(default_factory=list)


# ── Store access helper ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _pipeline():
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    return IndexPipeline()


@lru_cache(maxsize=1)
def _hybrid_retriever():
    from cce.retrieval.hybrid import HybridRetriever  # noqa: PLC0415
    from cce.config import get_settings  # noqa: PLC0415
    p = _pipeline()
    return HybridRetriever(
        sym_store=p.symbol_store,
        lex_store=p.lexical_store,
        graph_store=p.graph_store,
        settings=get_settings(),
    )


# ── Implemented (Phases 1-5) ──────────────────────────────────────────────────

def search_code(
    query: str,
    mode: Literal["auto", "lexical", "semantic", "hybrid"] = "auto",
    k: int = 10,
    filters: dict | None = None,
) -> list[Hit]:
    """Multi-mode code search.

    - lexical:  SQLite FTS5 symbol search only (fastest, always available)
    - hybrid:   Symbol BM25 + file BM25 + vector → RRF + graph expansion (Phase 8)
    - semantic: Qdrant vector only (requires Phase 7 index)
    - auto:     hybrid if index exists, else lexical
    """
    if mode == "semantic":
        return [_hr_to_hit(r) for r in _semantic_search(query, k, filters)]

    if mode == "lexical":
        p = _pipeline()
        hits: list[Hit] = []
        seen_paths: set[str] = set()
        for sh in p.symbol_store.search(query, k=k):
            hits.append(Hit(
                node=sh.node, path=sh.node.file_path,
                line_start=sh.node.line_start, line_end=sh.node.line_end,
                snippet=sh.node.signature or sh.node.name,
                score=abs(sh.rank), provenance="lex",
            ))
            seen_paths.add(sh.node.file_path)
        for lh in p.lexical_store.search(query, k=k):
            if lh.path not in seen_paths:
                hits.append(Hit(
                    node=None, path=lh.path,
                    line_start=0, line_end=0,
                    snippet=lh.snippet,
                    score=abs(lh.rank), provenance="lex",
                ))
        return hits[:k]

    # hybrid / auto — use HybridRetriever (Phase 8)
    try:
        results = _hybrid_retriever().retrieve(query, k=k, filters=filters)
        return [_hr_to_hit(r) for r in results]
    except Exception:  # noqa: BLE001
        # fallback to lexical if hybrid fails
        return search_code(query, mode="lexical", k=k, filters=filters)


def _hr_to_hit(r) -> Hit:
    prov_str = "|".join(sorted(r.provenance)) if r.provenance else "hybrid"
    prov_lit = "hybrid" if "|" in prov_str else (prov_str if prov_str in ("lex","vec","graph") else "hybrid")
    return Hit(
        node=r.node, path=r.path,
        line_start=r.line_start, line_end=r.line_end,
        snippet=r.snippet, score=r.rrf_score,
        provenance=prov_lit,  # type: ignore[arg-type]
    )


def get_symbol(qualified_name: str) -> Node:
    node = _pipeline().symbol_store.get_by_qname(qualified_name)
    if not node:
        raise KeyError(f"Symbol not found: {qualified_name}")
    return node


def get_file_outline(path: str) -> list[Node]:
    return _pipeline().symbol_store.get_for_file(path)


def find_references(qualified_name: str) -> list[Location]:
    p = _pipeline()
    node = p.symbol_store.get_by_qname(qualified_name)
    if not node:
        return []
    edges = p.graph_store.find_references(node.id)
    return [e.location for e in edges if e.location]


def find_callers(qualified_name: str) -> list[Node]:
    p = _pipeline()
    node = p.symbol_store.get_by_qname(qualified_name)
    if not node:
        return []
    return p.graph_store.find_callers(node.id)


def find_implementations(qualified_name: str) -> list[Node]:
    p = _pipeline()
    node = p.symbol_store.get_by_qname(qualified_name)
    if not node:
        return []
    return p.graph_store.find_implementations(node.id)


def get_neighborhood(
    qualified_name: str,
    depth: int = 2,
    edge_kinds: list[str] | None = None,
) -> SubGraph:
    p = _pipeline()
    node = p.symbol_store.get_by_qname(qualified_name)
    if not node:
        raise KeyError(f"Symbol not found: {qualified_name}")
    kinds = [EdgeKind(k) for k in (edge_kinds or [])] or None
    sg = p.graph_store.get_neighborhood(node.id, depth=depth, edge_kinds=kinds)
    return SubGraph(root_id=sg.root_id, nodes=sg.nodes, edges=sg.edges)


# ── Phase 6: framework-aware tools ────────────────────────────────────────────

def get_route(pattern_or_path: str) -> RouteInfo:
    """Resolve a URL path or pattern to its handler + response model.

    Searches NodeKind.ROUTE and NodeKind.URL_PATTERN nodes by name or meta.pattern.
    """
    p = _pipeline()
    conn = p.symbol_store._db.conn

    rows = conn.execute(
        "SELECT * FROM symbols WHERE kind IN ('Route','URLPattern') "
        "AND (name = ? OR json_extract(meta,'$.pattern') = ? OR json_extract(meta,'$.path') = ?)",
        (pattern_or_path, pattern_or_path, pattern_or_path),
    ).fetchall()

    if not rows:
        # partial match
        rows = conn.execute(
            "SELECT * FROM symbols WHERE kind IN ('Route','URLPattern') AND name LIKE ?",
            (f"%{pattern_or_path}%",),
        ).fetchall()

    if not rows:
        raise KeyError(f"Route not found: {pattern_or_path}")

    from cce.index.symbol_store import _row_to_node  # noqa: PLC0415

    node = _row_to_node(rows[0])
    meta = node.meta
    handler_qname = meta.get("handler", meta.get("view_ref", ""))
    methods = meta.get("methods", [])
    if isinstance(methods, str):
        methods = [methods]
    return RouteInfo(
        pattern=node.name,
        methods=methods,
        handler_qname=handler_qname,
        framework=node.framework_tag.value if node.framework_tag else "unknown",
        response_model=meta.get("response_model"),
    )


def get_component_tree(component_name: str) -> ComponentTree:
    """Return the render tree, hooks, and props of a React component."""
    p = _pipeline()
    conn = p.symbol_store._db.conn

    rows = conn.execute(
        "SELECT * FROM symbols WHERE kind = 'Component' AND name = ? LIMIT 1",
        (component_name,),
    ).fetchall()
    if not rows:
        raise KeyError(f"Component not found: {component_name}")

    from cce.index.symbol_store import _row_to_node  # noqa: PLC0415
    node = _row_to_node(rows[0])

    # RENDERS edges → children
    child_rows = conn.execute(
        "SELECT s.name FROM symbols s "
        "JOIN edges e ON s.id = e.dst_id "
        "WHERE e.src_id = ? AND e.kind = 'RENDERS'",
        (node.id,),
    ).fetchall()
    children = [r["name"] for r in child_rows]

    # USES_HOOK edges → hooks
    hook_rows = conn.execute(
        "SELECT e.* FROM edges e "
        "WHERE e.src_id = ? AND e.kind = 'USES_HOOK'",
        (node.id,),
    ).fetchall()
    hooks = [r["dst_id"] for r in hook_rows]

    props = node.meta.get("props", [])
    return ComponentTree(
        component_qname=node.qualified_name,
        children=children,
        hooks=hooks,
        props=props,
    )


def get_api_flow(route_or_component: str) -> CrossStackFlow:
    """Return the UI → API → handler → model path for a route or component.

    Path matching algorithm:
    1. Try exact match on the given string.
    2. Normalise the input URL (strip /api/v1 prefixes, replace numeric and UUID
       segments with {id}-style templates) and match against all Route nodes.
    3. Walk ROUTES_TO → handler → response_model chain.
    """
    p = _pipeline()
    conn = p.symbol_store._db.conn
    steps: list[dict] = []

    # ── Step 1: find the matching route node ──────────────────────────────────
    route: RouteInfo | None = None
    try:
        route = get_route(route_or_component)
    except KeyError:
        # ── Step 2: normalise + template-match ────────────────────────────────
        normalized = _normalize_api_path(route_or_component)
        rows = conn.execute(
            "SELECT * FROM symbols WHERE kind IN ('Route','URLPattern')",
        ).fetchall()
        from cce.index.symbol_store import _row_to_node  # noqa: PLC0415
        for row in rows:
            node = _row_to_node(row)
            pattern = node.meta.get("path") or node.meta.get("pattern") or node.name
            if _path_templates_match(normalized, pattern):
                meta = node.meta
                route = RouteInfo(
                    pattern=pattern,
                    methods=meta.get("methods", []),
                    handler_qname=meta.get("handler", meta.get("view_ref", "")),
                    framework=node.framework_tag.value if node.framework_tag else "unknown",
                    response_model=meta.get("response_model"),
                )
                break

    if route:
        steps.append({"kind": "Route", "name": route.pattern, "framework": route.framework,
                      "methods": route.methods})
        if route.handler_qname:
            handler = p.symbol_store.get_by_qname(route.handler_qname)
            if handler:
                steps.append({"kind": "Handler", "name": handler.name,
                               "file": handler.file_path, "line": handler.line_start})
                if route.response_model:
                    from cce.index.symbol_store import _row_to_node  # noqa: PLC0415
                    # Peel off common generic wrappers so `list[UserResponse]`
                    # and `Optional[UserResponse]` still resolve to the model.
                    candidates = _extract_model_candidates(route.response_model)
                    model = None
                    for cand in candidates:
                        model = p.symbol_store.get_by_qname(cand)
                        if model:
                            break
                        rows = conn.execute(
                            "SELECT * FROM symbols WHERE name = ? LIMIT 1", (cand,),
                        ).fetchall()
                        if rows:
                            model = _row_to_node(rows[0])
                            break
                    if model:
                        steps.append({"kind": model.kind.value, "name": model.name,
                                      "file": model.file_path})

    return CrossStackFlow(anchor=route_or_component, steps=steps)


# ── Path normalisation helpers ────────────────────────────────────────────────

import re as _re


def _extract_model_candidates(type_str: str) -> list[str]:
    """Return likely model names from a type annotation.

    ``"list[UserResponse]"`` → ``["list[UserResponse]", "UserResponse"]``
    ``"Optional[User]"``     → ``["Optional[User]", "User"]``
    ``"Union[A, B]"``        → ``["Union[A, B]", "A", "B"]``
    """
    out = [type_str]
    inner = _re.findall(r"\[([^\[\]]+)\]", type_str)
    for group in inner:
        for part in group.split(","):
            name = part.strip()
            if name and name not in out:
                out.append(name)
    return out

# Patterns that indicate a path segment is a concrete value (not a template)
_ID_PATTERNS = [
    _re.compile(r"^\d+$"),                                       # 42
    _re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),  # UUID
    _re.compile(r"^[0-9A-Za-z]{20,}$"),                          # ULID / slug
]
_API_PREFIX = _re.compile(r"^/?(api/v\d+/|api/)")


def _normalize_api_path(url: str) -> str:
    """Strip /api/v1 prefix and replace concrete id segments with {id}."""
    # Remove query string
    url = url.split("?")[0].rstrip("/")
    # Strip /api/v1/, /api/ etc.
    url = _API_PREFIX.sub("/", url)
    segments = [s for s in url.split("/") if s]
    normalized = []
    for seg in segments:
        if any(p.match(seg) for p in _ID_PATTERNS):
            normalized.append("{id}")
        else:
            normalized.append(seg)
    return "/" + "/".join(normalized)


def _path_templates_match(concrete: str, template: str) -> bool:
    """Return True if *concrete* (normalised) matches *template* (FastAPI/Django pattern).

    Converts FastAPI {param} and Django <type:name> to wildcards, then compares.
    """
    # Normalise both sides
    t = _re.sub(r"\{[^}]+\}", "{id}", template.rstrip("/") or "/")
    t = _re.sub(r"<[^>]+>", "{id}", t)   # Django <int:pk>
    t = _re.sub(r":\w+", "", t)           # Express :id → bare segment
    c = concrete.rstrip("/") or "/"
    return c == t


# ── Phase 7: semantic search ────────────────────────────────────────────────────

def _semantic_search(query: str, k: int, filters: dict | None) -> list[Hit]:
    """Query Qdrant with nomic-embed-code embeddings."""
    from cce.config import get_settings  # noqa: PLC0415
    from cce.embeddings.embedder import get_embedder  # noqa: PLC0415
    from cce.index.vector_store import VectorStore  # noqa: PLC0415

    settings = get_settings()
    embedder = get_embedder()
    vstore = VectorStore(settings)

    # We need to know the root for collection naming; use stored files
    conn = _pipeline().symbol_store._db.conn
    sample = conn.execute("SELECT path FROM files LIMIT 1").fetchone()
    if not sample:
        return []

    # Derive root from any indexed file — store collection per DB path instead
    collection = vstore.collection_name_from_db(settings.paths.sqlite_db)
    if not vstore.collection_exists(collection):
        return []

    query_vec = embedder.embed_query(query)
    results = vstore.search(collection, query_vec, k=k, filters=filters)

    hits: list[Hit] = []
    for r in results:
        payload = r.payload or {}
        node_id = payload.get("node_id", "")
        node = _pipeline().graph_store.get_node(node_id) if node_id else None
        hits.append(Hit(
            node=node,
            path=payload.get("path", ""),
            line_start=node.line_start if node else 0,
            line_end=node.line_end if node else 0,
            snippet=payload.get("header", ""),
            score=r.score,
            provenance="vec",
        ))
    return hits
