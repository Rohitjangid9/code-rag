"""Phase 12 — Eval harness, metrics, and dataset tests."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from cce.eval.metrics import (
    dcg_at_k, ideal_dcg_at_k, mean_ndcg_at_k, mean_recall_at_k,
    mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PY = FIXTURES / "sample_python"


# ── Metrics: reciprocal rank ───────────────────────────────────────────────────

def test_rr_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_rr_second_hit():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)


def test_rr_no_hit():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_rr_empty_ranked():
    assert reciprocal_rank([], {"a"}) == 0.0


# ── Metrics: recall@k ─────────────────────────────────────────────────────────

def test_recall_all_found():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == pytest.approx(0.5)


def test_recall_k_truncates():
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0  # a is rank 3


def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=5) == 1.0


# ── Metrics: nDCG ────────────────────────────────────────────────────────────

def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_no_hits():
    assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_dcg_decreases_with_rank():
    dcg1 = dcg_at_k(["a", "b"], {"a"}, k=2)
    dcg2 = dcg_at_k(["b", "a"], {"a"}, k=2)
    assert dcg1 > dcg2


def test_ideal_dcg_at_k_bounds():
    idcg = ideal_dcg_at_k({"a", "b", "c"}, k=3)
    assert idcg == pytest.approx(sum(1.0 / math.log2(i + 2) for i in range(3)))


# ── Metrics: aggregates ────────────────────────────────────────────────────────

def test_mrr_at_k_all_first():
    results = [(["a", "b"], {"a"}), (["c", "d"], {"c"})]
    assert mrr_at_k(results, k=2) == 1.0


def test_mrr_at_k_mixed():
    results = [(["a", "b"], {"b"}), (["c", "d"], {"c"})]
    assert mrr_at_k(results, k=2) == pytest.approx(0.75)


def test_mean_recall_at_k():
    results = [(["a", "b", "c"], {"a", "b"}), (["x", "y"], {"x"})]
    assert mean_recall_at_k(results, k=3) == 1.0


def test_mean_ndcg_perfect():
    results = [(["a"], {"a"}), (["b"], {"b"})]
    assert mean_ndcg_at_k(results, k=1) == pytest.approx(1.0)


def test_metrics_empty_inputs():
    assert mrr_at_k([], k=10) == 0.0
    assert mean_recall_at_k([], k=10) == 0.0
    assert mean_ndcg_at_k([], k=10) == 0.0


# ── EvalDataset ───────────────────────────────────────────────────────────────

def test_load_eval_dataset():
    from cce.eval.dataset import EvalDataset  # noqa: PLC0415
    ds = EvalDataset.from_yaml(FIXTURES / "eval_queries.yaml")
    assert len(ds) >= 5
    assert all(q.query for q in ds.queries)
    assert all(q.id for q in ds.queries)


def test_dataset_expected_symbols_populated():
    from cce.eval.dataset import EvalDataset  # noqa: PLC0415
    ds = EvalDataset.from_yaml(FIXTURES / "eval_queries.yaml")
    assert any(len(q.expected_symbols) > 0 for q in ds.queries)


# ── EvalHarness (mocked retriever) ────────────────────────────────────────────

def _make_mock_hit(qname: str, path: str = "models.py"):
    node = type("N", (), {"qualified_name": qname, "file_path": path,
                          "line_start": 1, "line_end": 5})()
    hit = type("H", (), {"node": node, "path": path, "line_start": 1,
                         "line_end": 5, "snippet": qname, "score": 0.9,
                         "provenance": "lex"})()
    return hit


def test_harness_runs_all_queries(tmp_path):
    from cce.eval.dataset import EvalDataset  # noqa: PLC0415
    from cce.eval.harness import EvalHarness  # noqa: PLC0415

    ds = EvalDataset.from_yaml(FIXTURES / "eval_queries.yaml")

    def fake_search(query, mode="hybrid", k=10, filters=None):
        return [_make_mock_hit("models.User"), _make_mock_hit("models.AdminUser")]

    with patch("cce.eval.harness.search_code", side_effect=fake_search):
        harness = EvalHarness(root=SAMPLE_PY, k=10)
        report = harness.run(ds)

    assert report.total_queries == len(ds)
    assert 0.0 <= report.mrr <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.ndcg <= 1.0


def test_harness_report_rich_table(tmp_path):
    from cce.eval.dataset import EvalDataset  # noqa: PLC0415
    from cce.eval.harness import EvalHarness  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    ds = EvalDataset.from_yaml(FIXTURES / "eval_queries.yaml")
    with patch("cce.eval.harness.search_code", return_value=[]):
        harness = EvalHarness(root=SAMPLE_PY, k=10)
        report = harness.run(ds)

    tbl = report.rich_table()
    assert isinstance(tbl, Table)


def test_harness_as_dict_schema():
    from cce.eval.harness import EvalReport  # noqa: PLC0415

    report = EvalReport(k=10, mrr=0.75, recall=0.80, ndcg=0.78, precision=0.60, total_queries=5)
    d = report.as_dict()
    assert set(d.keys()) >= {"k", "mrr", "recall", "ndcg", "total_queries"}
    assert d["mrr"] == 0.75
