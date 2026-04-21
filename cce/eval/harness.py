"""Phase 12 — EvalHarness: run retrieval over an EvalDataset and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Table

from cce.eval.dataset import EvalDataset, EvalQuery
from cce.eval.metrics import mean_ndcg_at_k, mean_recall_at_k, mrr_at_k
from cce.logging import get_logger

log = get_logger(__name__)


@dataclass
class QueryResult:
    query_id: str
    query: str
    ranked_symbols: list[str]
    ranked_files: list[str]
    expected_symbols: set[str]
    expected_files: set[str]


@dataclass
class EvalReport:
    k: int
    mrr: float
    recall: float
    ndcg: float
    precision: float
    query_results: list[QueryResult] = field(default_factory=list)
    total_queries: int = 0

    def rich_table(self) -> Table:
        tbl = Table(title=f"Retrieval Eval @ k={self.k}", show_lines=True)
        tbl.add_column("Metric", style="cyan", width=14)
        tbl.add_column("Score", justify="right", style="bold")
        tbl.add_row("MRR@k",    f"{self.mrr:.4f}")
        tbl.add_row("Recall@k", f"{self.recall:.4f}")
        tbl.add_row("nDCG@k",   f"{self.ndcg:.4f}")
        tbl.add_row("Queries",  str(self.total_queries))
        return tbl

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "mrr": round(self.mrr, 4),
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
            "total_queries": self.total_queries,
        }


class EvalHarness:
    """Runs hybrid retrieval for each query and computes aggregate metrics."""

    def __init__(self, root: Path, k: int = 10, mode: str = "hybrid") -> None:
        self._root = root
        self._k = k
        self._mode = mode

    def run(self, dataset: EvalDataset) -> EvalReport:
        """Evaluate all queries in *dataset* and return an EvalReport."""
        from cce.retrieval.tools import search_code  # noqa: PLC0415

        query_results: list[QueryResult] = []
        sym_pairs: list[tuple[list[str], set[str]]] = []
        file_pairs: list[tuple[list[str], set[str]]] = []

        for eq in dataset.queries:
            log.info("Eval query: %s", eq.id)
            try:
                hits = search_code(eq.query, mode=self._mode, k=self._k)
            except Exception as exc:  # noqa: BLE001
                log.warning("search_code failed for %s: %s", eq.id, exc)
                hits = []

            ranked_syms = [h.node.qualified_name for h in hits if h.node]
            ranked_files = list(dict.fromkeys(h.path for h in hits))  # ordered unique

            qr = QueryResult(
                query_id=eq.id,
                query=eq.query,
                ranked_symbols=ranked_syms,
                ranked_files=ranked_files,
                expected_symbols=set(eq.expected_symbols),
                expected_files=set(eq.expected_files),
            )
            query_results.append(qr)
            sym_pairs.append((ranked_syms, set(eq.expected_symbols)))
            file_pairs.append((ranked_files, set(eq.expected_files)))

        k = self._k
        mrr   = mrr_at_k(sym_pairs, k)
        recall = mean_recall_at_k(sym_pairs, k)
        ndcg  = mean_ndcg_at_k(sym_pairs, k)

        return EvalReport(
            k=k,
            mrr=mrr,
            recall=recall,
            ndcg=ndcg,
            precision=0.0,  # computed per-query if needed
            query_results=query_results,
            total_queries=len(dataset),
        )
