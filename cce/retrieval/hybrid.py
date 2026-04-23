"""Phase 8 — Hybrid Retriever: Symbol BM25 + Vector + RRF + Graph Expansion.

Pipeline (F19: synonym expansion; F20: optional cross-encoder reranker; F25: MMR):
  1. F19: Expand query via synonym map (auth ↔ authenticate/oauth/jwt …)
  2. Symbol FTS BM25 (top 50)   — name/qname/docstring/signature
  3. File-content BM25 (top 30) → expand to symbols (top 3 per file)
  4. Vector cosine (top 50)     — Qdrant, graceful fallback
  5. Reciprocal Rank Fusion (k=60) over node-id space
  6. Graph-expand top 15 via CALLS/INHERITS/ROUTES_TO/RENDERS (1 hop)
  7. F20: Optional cross-encoder reranker (top 50 candidates, CCE_RETRIEVAL__RERANK=true)
  8. F25: MMR diversity (λ=0.6) over top 2·k candidates
  9. Deduplicate + return top k with provenance labels
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cce.embeddings.chunker import build_header
from cce.graph.schema import EdgeKind, Node
from cce.logging import get_logger

if TYPE_CHECKING:
    from cce.config import Settings
    from cce.graph.sqlite_store import SQLiteGraphStore
    from cce.index.lexical_store import LexicalStore
    from cce.index.symbol_store import SymbolStore

log = get_logger(__name__)

_EXPAND_EDGE_KINDS = [EdgeKind.CALLS, EdgeKind.INHERITS, EdgeKind.ROUTES_TO, EdgeKind.RENDERS]

# ── F19: synonym map ─────────────────────────────────────────────────────────
# Keys are query tokens; values are additional tokens to OR into the query.
_SYNONYMS: dict[str, list[str]] = {
    "auth":         ["authenticate", "authorization", "authorize", "oauth", "jwt",
                     "session", "bearer", "login", "token", "credential"],
    "authenticate": ["auth", "login", "oauth", "jwt", "bearer"],
    "authorize":    ["auth", "permission", "rbac", "acl", "role"],
    "db":           ["database", "sql", "sqlite", "postgres", "postgresql", "orm",
                     "repository", "model", "schema"],
    "database":     ["db", "sql", "sqlite", "postgres", "orm"],
    "api":          ["endpoint", "route", "handler", "view", "controller"],
    "route":        ["endpoint", "url", "path", "handler", "view"],
    "config":       ["settings", "configuration", "env", "environment"],
    "settings":     ["config", "configuration", "env"],
    "log":          ["logging", "logger", "trace"],
    "logging":      ["log", "logger", "trace"],
    "error":        ["exception", "raise", "traceback", "failure"],
    "test":         ["spec", "unit", "pytest", "assert"],
    "cache":        ["redis", "memcache", "lru", "ttl"],
    "embed":        ["embedding", "vector", "encode", "qdrant"],
    "search":       ["retrieval", "query", "bm25", "fts", "semantic"],
}


def _expand_query(query: str) -> str:
    """Expand query tokens using the synonym map (F19).

    Appends synonyms as OR-joined tokens to the original query so BM25 hits
    on semantically related tokens.  Only expands when the token is an exact
    key in the synonym map (case-insensitive).
    """
    tokens = [t.lower() for t in query.split() if len(t) > 1]
    extra: list[str] = []
    for token in tokens:
        for syn in _SYNONYMS.get(token, []):
            if syn not in tokens and syn not in extra:
                extra.append(syn)
    if not extra:
        return query
    return query + " " + " ".join(extra)


@dataclass
class HybridResult:
    node_id: str
    node: Node | None
    path: str
    line_start: int
    line_end: int
    snippet: str
    rrf_score: float
    provenance: list[str] = field(default_factory=list)   # ["lex", "vec", "graph"]
    chunk_header: str = ""


def _mmr(
    candidates: list["HybridResult"],
    k: int,
    lam: float = 0.6,
) -> list["HybridResult"]:
    """Maximal Marginal Relevance diversity filter (F25).

    Re-ranks *candidates* by balancing relevance (rrf_score) with diversity
    (cosine distance between chunk headers).  Returns top-*k* diverse results.
    λ=1.0 → pure relevance; λ=0.0 → pure diversity.
    """
    if len(candidates) <= k:
        return candidates

    # Simple token-overlap similarity (no heavy embedding dependency)
    def _sim(a: "HybridResult", b: "HybridResult") -> float:
        ta = set(a.snippet.lower().split())
        tb = set(b.snippet.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    selected: list["HybridResult"] = []
    remaining = list(candidates)

    while remaining and len(selected) < k:
        if not selected:
            # Bootstrap: pick the highest-scoring candidate
            best = max(remaining, key=lambda r: r.rrf_score)
        else:
            best = max(
                remaining,
                key=lambda r: (
                    lam * r.rrf_score
                    - (1.0 - lam) * max(_sim(r, s) for s in selected)
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


def _rrf_merge(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of node-ids."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, node_id in enumerate(ranking):
            scores[node_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    """Fuses BM25, vector, and graph signals into a single ranked result list."""

    def __init__(
        self,
        sym_store: "SymbolStore",
        lex_store: "LexicalStore",
        graph_store: "SQLiteGraphStore",
        settings: "Settings",
    ) -> None:
        self._sym = sym_store
        self._lex = lex_store
        self._graph = graph_store
        self._settings = settings
        self._embedder = None
        self._vstore = None

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10, filters: dict | None = None) -> list[HybridResult]:
        """Run hybrid retrieval and return top-k results."""
        from cce.config import get_settings  # noqa: PLC0415
        cfg = get_settings().retrieval

        # 1. F19: synonym expansion
        expanded_query = _expand_query(query) if cfg.synonym_expansion else query

        # 2. Symbol FTS BM25
        sym_hits = self._sym.search(expanded_query, k=50)
        sym_ranking = [h.node.id for h in sym_hits]

        # 3. File-content BM25 → expand to symbol ids
        lex_ranking = self._lex_to_symbol_ids(expanded_query, k=30)

        # 4. Vector (graceful fallback) — use original query for better semantic match
        vec_ranking = self._vector_search(query, k=50, filters=filters)

        # 5. RRF merge
        rankings = [r for r in [sym_ranking, lex_ranking, vec_ranking] if r]
        if not rankings:
            return []
        merged = _rrf_merge(rankings)

        # Build provenance sets for top results
        sym_set = set(sym_ranking)
        vec_set = set(vec_ranking)

        # 6. Hydrate top (k + expansion headroom)
        results: dict[str, HybridResult] = {}
        for node_id, score in merged:
            node = self._graph.get_node(node_id)
            if not node:
                continue
            prov = []
            if node_id in sym_set:
                prov.append("lex")
            if node_id in vec_set:
                prov.append("vec")
            # F-BODY: seed snippet with symbol body from lex_sym_fts so the
            # reasoner's has_symbol_body axis (len > 50) is satisfied without
            # an extra get_file_slice() round-trip.
            body_snippet = self._fetch_body_snippet(node.qualified_name)
            snippet = body_snippet or node.signature or node.name
            results[node_id] = HybridResult(
                node_id=node_id, node=node,
                path=node.file_path,
                line_start=node.line_start, line_end=node.line_end,
                snippet=snippet,
                rrf_score=score,
                provenance=prov,
                chunk_header=build_header(node),
            )

        # 7. Graph expand top 15
        top15 = [nid for nid, _ in merged[:15]]
        self._graph_expand(top15, results)

        # 8. Deduplicate, sort
        seen: set[str] = set()
        candidates: list[HybridResult] = []
        for r in sorted(results.values(), key=lambda r: -r.rrf_score):
            dedup_key = (r.path, r.line_start)
            if dedup_key not in seen:
                seen.add(dedup_key)
                candidates.append(r)

        # 9. F20: optional cross-encoder reranker
        if cfg.rerank and len(candidates) > k:
            candidates = self._rerank(query, candidates, cfg)

        # 10. F25: MMR diversity filter
        final = _mmr(candidates[: k * 2], k=k)
        return final

    # ── F20: cross-encoder reranker ───────────────────────────────────────────

    def _rerank(
        self, query: str, candidates: list[HybridResult], cfg: "Any",
    ) -> list[HybridResult]:
        """Re-rank candidates using a cross-encoder model (F20).

        Requires ``sentence-transformers``.  Falls back silently to the
        original ranking if the package is unavailable.
        """
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError:
            log.debug("cross-encoder reranker requested but sentence-transformers not installed; skipping")
            return candidates

        top = candidates[: cfg.rerank_top_n]
        pairs = [(query, r.snippet or r.chunk_header or "") for r in top]
        try:
            model = CrossEncoder(cfg.rerank_model)
            scores = model.predict(pairs)
            reranked = sorted(zip(top, scores), key=lambda t: -t[1])
            top_reranked = [r for r, _ in reranked]
        except Exception as exc:  # noqa: BLE001
            log.warning("Cross-encoder reranker failed: %s — using original order", exc)
            top_reranked = top

        # Append any candidates beyond rerank_top_n unchanged
        return top_reranked + candidates[cfg.rerank_top_n:]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _lex_to_symbol_ids(self, query: str, k: int) -> list[str]:
        """File-content BM25 → top-3 symbols per file hit."""
        node_ids: list[str] = []
        seen: set[str] = set()
        for lh in self._lex.search(query, k=k):
            for sym in self._sym.get_for_file(lh.path)[:3]:
                if sym.id not in seen:
                    seen.add(sym.id)
                    node_ids.append(sym.id)
        return node_ids

    def _vector_search(self, query: str, k: int, filters: dict | None) -> list[str]:
        """Return ranked node-ids from Qdrant; empty list if unavailable."""
        try:
            if self._vstore is None:
                from cce.index.vector_store import VectorStore  # noqa: PLC0415
                self._vstore = VectorStore(self._settings)
            if self._embedder is None:
                from cce.embeddings.embedder import get_embedder  # noqa: PLC0415
                self._embedder = get_embedder()
            # F-M3: derive collection from repo root (single source of truth
            # shared with the indexer).  Fall back to data_dir.parent when no
            # explicit root is bound — matches the ``<root>/.cce`` convention.
            root = self._settings.repo_root
            if root is None:
                root = self._settings.paths.data_dir.resolve().parent
            collection = self._vstore.collection_name(root)
            if not self._vstore.collection_exists(collection):
                return []
            query_vec = self._embedder.embed_query(query)
            results = self._vstore.search(collection, query_vec, k=k, filters=filters)
            return [r.payload.get("node_id", "") for r in results if r.payload]
        except Exception as exc:  # noqa: BLE001
            log.debug("Vector search skipped: %s", exc)
            return []

    def _fetch_body_snippet(self, qualified_name: str) -> str:
        """Return the stored symbol body from lex_sym_fts (up to 400 chars).

        Falls back to ``""`` when no record is found so callers can safely
        fall through to the signature-based snippet.
        """
        try:
            row = self._lex._db.conn.execute(
                "SELECT content FROM lex_sym_fts WHERE qualified_name = ? LIMIT 1",
                (qualified_name,),
            ).fetchone()
            if row:
                return row["content"][:400]
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _graph_expand(self, top_ids: list[str], results: dict[str, HybridResult]) -> None:
        """1-hop expansion; adds neighbors to *results* in-place with provenance='graph'."""
        for node_id in top_ids:
            try:
                sg = self._graph.get_neighborhood(node_id, depth=1, edge_kinds=_EXPAND_EDGE_KINDS)
            except Exception:  # noqa: BLE001
                continue
            for neighbor in sg.nodes:
                if neighbor.id not in results:
                    body = self._fetch_body_snippet(neighbor.qualified_name)
                    results[neighbor.id] = HybridResult(
                        node_id=neighbor.id, node=neighbor,
                        path=neighbor.file_path,
                        line_start=neighbor.line_start, line_end=neighbor.line_end,
                        snippet=body or neighbor.signature or neighbor.name,
                        rrf_score=0.05,   # below any directly-retrieved result
                        provenance=["graph"],
                        chunk_header=build_header(neighbor),
                    )
                elif "graph" not in results[neighbor.id].provenance:
                    results[neighbor.id].provenance.append("graph")
