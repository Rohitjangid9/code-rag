"""Phase 12 — EvalHarness: run retrieval over an EvalDataset and aggregate metrics.

F14: writes a ``scorecard.json`` after every run so CI pipelines can assert
thresholds with a simple script or jq query.

Score card format::

    {
        "dataset": "<name>",
        "k": 10,
        "mrr": 0.85,
        "recall": 0.90,
        "ndcg": 0.88,
        "pass": true
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Table

from cce.eval.dataset import EvalDataset, EvalQuery
from cce.eval.metrics import mean_ndcg_at_k, mean_recall_at_k, mrr_at_k
from cce.logging import get_logger
from cce.retrieval.tools import search_code

log = get_logger(__name__)

# Default CI gate thresholds (overridable via ``EvalHarness(thresholds=…)``)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "mrr": 0.50,
    "recall": 0.50,
    "ndcg": 0.50,
}


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

    def as_dict(self, dataset_name: str = "") -> dict:
        return {
            "dataset": dataset_name,
            "k": self.k,
            "mrr": round(self.mrr, 4),
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
            "total_queries": self.total_queries,
        }

    def write_scorecard(
        self,
        path: Path,
        dataset_name: str = "",
        thresholds: dict | None = None,
    ) -> bool:
        """Write a ``scorecard.json`` and return True if all thresholds pass (F14)."""
        th = thresholds or DEFAULT_THRESHOLDS
        passed = (
            self.mrr >= th.get("mrr", 0.0)
            and self.recall >= th.get("recall", 0.0)
            and self.ndcg >= th.get("ndcg", 0.0)
        )
        scorecard = {**self.as_dict(dataset_name), "pass": passed, "thresholds": th}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        log.info("Scorecard written to %s (pass=%s)", path, passed)
        return passed


class EvalHarness:
    """Runs hybrid retrieval for each query and computes aggregate metrics."""

    def __init__(
        self,
        root: Path,
        k: int = 10,
        mode: str = "hybrid",
        thresholds: dict | None = None,
    ) -> None:
        self._root = root
        self._k = k
        self._mode = mode
        self._thresholds = thresholds or DEFAULT_THRESHOLDS

    def run(self, dataset: EvalDataset) -> EvalReport:
        """Evaluate all queries in *dataset* and return an EvalReport."""
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


# ── F-M9: generic repo-agnostic sanity checks ────────────────────────────────

@dataclass
class GenericEvalResult:
    """Per-check outcome for :func:`run_generic_eval`."""

    check: str
    passed: bool
    detail: str = ""


def run_generic_eval(repo_root: Path | None = None) -> list[GenericEvalResult]:
    """Run repo-agnostic retrieval sanity checks.

    These tests do not use expected symbols — they just verify that the
    indexed data is reachable through the tool surface.  Useful for
    validating a fresh index on an unknown codebase.
    """
    from cce.config import get_settings  # noqa: PLC0415
    from cce.retrieval.tools import (  # noqa: PLC0415
        list_files,
        list_routes,
        search_code,
    )

    results: list[GenericEvalResult] = []
    settings = get_settings(repo_root=repo_root) if repo_root else get_settings()

    # 1. File enumeration must return at least one file.
    try:
        files = list_files(limit=50)
        results.append(GenericEvalResult(
            check="list_files",
            passed=len(files) > 0,
            detail=f"{len(files)} files returned",
        ))
    except Exception as exc:  # noqa: BLE001
        results.append(GenericEvalResult("list_files", False, f"error: {exc}"))

    # 2. Lexical search over a common token should return something.
    for token in ("def", "function", "class", "return"):
        try:
            hits = search_code(token, mode="lexical", k=5)
            if hits:
                results.append(GenericEvalResult(
                    check="search_code.lexical",
                    passed=True,
                    detail=f"token={token!r} → {len(hits)} hits",
                ))
                break
        except Exception as exc:  # noqa: BLE001
            results.append(GenericEvalResult(
                "search_code.lexical", False, f"error: {exc}",
            ))
            break
    else:
        results.append(GenericEvalResult(
            "search_code.lexical", False, "no hits for any common token",
        ))

    # 3. If a web framework is present in the manifest, list_routes must work.
    manifest_path = settings.paths.data_dir / "index.json"
    frameworks: list[str] = []
    if manifest_path.exists():
        try:
            frameworks = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "frameworks", []
            )
        except Exception:  # noqa: BLE001
            frameworks = []

    web_frameworks = {"fastapi", "django", "drf"}
    if any(f in web_frameworks for f in frameworks):
        try:
            routes = list_routes()
            results.append(GenericEvalResult(
                check="list_routes",
                passed=len(routes) > 0,
                detail=f"{len(routes)} routes returned",
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(GenericEvalResult("list_routes", False, f"error: {exc}"))
    else:
        results.append(GenericEvalResult(
            "list_routes", True, "skipped (no web framework detected)",
        ))

    return results
