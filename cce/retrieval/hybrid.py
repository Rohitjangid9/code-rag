"""Phase 8 — Hybrid Retriever: Symbol BM25 + Vector + RRF + Graph Expansion.

Pipeline (no reranker in v1):
  1. Symbol FTS BM25 (top 50)   — name/qname/docstring/signature
  2. File-content BM25 (top 30) → expand to symbols (top 3 per file)
  3. Vector cosine (top 50)     — Qdrant, graceful fallback
  4. Reciprocal Rank Fusion (k=60) over node-id space
  5. Graph-expand top 15 via CALLS/INHERITS/ROUTES_TO/RENDERS (1 hop)
  6. Deduplicate + return top k with provenance labels
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
        # 1. Symbol FTS BM25
        sym_hits = self._sym.search(query, k=50)
        sym_ranking = [h.node.id for h in sym_hits]

        # 2. File-content BM25 → expand to symbol ids
        lex_ranking = self._lex_to_symbol_ids(query, k=30)

        # 3. Vector (graceful fallback)
        vec_ranking = self._vector_search(query, k=50, filters=filters)

        # 4. RRF merge
        rankings = [r for r in [sym_ranking, lex_ranking, vec_ranking] if r]
        if not rankings:
            return []
        merged = _rrf_merge(rankings)

        # Build provenance sets for top results
        sym_set = set(sym_ranking)
        lex_set = set(lex_ranking)
        vec_set = set(vec_ranking)

        # 5. Hydrate top (k + expansion headroom)
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
            results[node_id] = HybridResult(
                node_id=node_id, node=node,
                path=node.file_path,
                line_start=node.line_start, line_end=node.line_end,
                snippet=node.signature or node.name,
                rrf_score=score,
                provenance=prov,
                chunk_header=build_header(node),
            )

        # 6. Graph expand top 15
        top15 = [nid for nid, _ in merged[:15]]
        self._graph_expand(top15, results)

        # 7. Deduplicate, sort, return top k
        seen: set[str] = set()
        final: list[HybridResult] = []
        for r in sorted(results.values(), key=lambda r: -r.rrf_score):
            dedup_key = (r.path, r.line_start)
            if dedup_key not in seen:
                seen.add(dedup_key)
                final.append(r)
        return final[:k]

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
            collection = self._vstore.collection_name_from_db(self._settings.paths.sqlite_db)
            if not self._vstore.collection_exists(collection):
                return []
            query_vec = self._embedder.embed_query(query)
            results = self._vstore.search(collection, query_vec, k=k, filters=filters)
            return [r.payload.get("node_id", "") for r in results if r.payload]
        except Exception as exc:  # noqa: BLE001
            log.debug("Vector search skipped: %s", exc)
            return []

    def _graph_expand(self, top_ids: list[str], results: dict[str, HybridResult]) -> None:
        """1-hop expansion; adds neighbors to *results* in-place with provenance='graph'."""
        for node_id in top_ids:
            try:
                sg = self._graph.get_neighborhood(node_id, depth=1, edge_kinds=_EXPAND_EDGE_KINDS)
            except Exception:  # noqa: BLE001
                continue
            for neighbor in sg.nodes:
                if neighbor.id not in results:
                    results[neighbor.id] = HybridResult(
                        node_id=neighbor.id, node=neighbor,
                        path=neighbor.file_path,
                        line_start=neighbor.line_start, line_end=neighbor.line_end,
                        snippet=neighbor.signature or neighbor.name,
                        rrf_score=0.05,   # below any directly-retrieved result
                        provenance=["graph"],
                        chunk_header=build_header(neighbor),
                    )
                elif "graph" not in results[neighbor.id].provenance:
                    results[neighbor.id].provenance.append("graph")
