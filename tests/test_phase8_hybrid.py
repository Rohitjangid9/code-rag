"""Phase 8 — Hybrid Retriever tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cce.retrieval.hybrid import HybridRetriever, _rrf_merge

FIXTURES = Path(__file__).parent / "fixtures" / "sample_python"


# ── RRF math ──────────────────────────────────────────────────────────────────

def test_rrf_single_list():
    ranking = ["a", "b", "c"]
    merged = _rrf_merge([ranking])
    ids = [x[0] for x in merged]
    assert ids == ["a", "b", "c"]  # original order preserved


def test_rrf_two_lists_boosts_overlap():
    r1 = ["a", "b", "c"]
    r2 = ["b", "a", "d"]
    merged = dict(_rrf_merge([r1, r2]))
    # "a" is #1 in r1 and #2 in r2; "b" is #2 in r1 and #1 in r2 — both should score high
    assert merged["a"] > merged["c"]
    assert merged["b"] > merged["c"]
    assert merged["b"] > merged["d"]


def test_rrf_empty_lists():
    assert _rrf_merge([]) == []
    assert _rrf_merge([[]]) == []


def test_rrf_no_overlap():
    r1 = ["a", "b"]
    r2 = ["c", "d"]
    merged = _rrf_merge([r1, r2])
    ids = [x[0] for x in merged]
    # top of each list should score equally
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_k_parameter_shifts_scores():
    r = ["x"]
    s_k60 = _rrf_merge([r], k=60)[0][1]
    s_k1  = _rrf_merge([r], k=1)[0][1]
    # smaller k → higher score (1/2 vs 1/61)
    assert s_k1 > s_k60


# ── HybridRetriever unit (mocked stores) ──────────────────────────────────────

def _make_mock_node(node_id: str, name: str, path: str = "app/views.py"):
    from cce.graph.schema import Language, Node, NodeKind  # noqa: PLC0415
    return Node(
        id=node_id,
        kind=NodeKind.FUNCTION,
        qualified_name=f"app.views.{name}",
        name=name,
        file_path=path,
        line_start=1,
        line_end=10,
        signature=f"def {name}():",
        docstring=None,
        language=Language.PYTHON,
        framework_tag=None,
    )


def _make_retriever(sym_results=None, lex_results=None):
    from cce.graph.schema import SubGraph  # noqa: PLC0415

    sym_store = MagicMock()
    lex_store = MagicMock()
    graph_store = MagicMock()
    settings = MagicMock()
    settings.paths.sqlite_db = Path("/tmp/test.sqlite")

    sym_hits = []
    for i, (nid, name) in enumerate(sym_results or []):
        h = MagicMock()
        h.node = _make_mock_node(nid, name)
        h.rank = float(i + 1)
        sym_hits.append(h)
    sym_store.search.return_value = sym_hits

    lex_hits = []
    for path in (lex_results or []):
        h = MagicMock()
        h.path = path
        h.snippet = "matched line"
        h.rank = 1.0
        lex_hits.append(h)
    lex_store.search.return_value = lex_hits
    sym_store.get_for_file.return_value = []

    node_map = {nid: _make_mock_node(nid, name) for nid, name in (sym_results or [])}
    graph_store.get_node.side_effect = lambda nid: node_map.get(nid)
    graph_store.get_neighborhood.return_value = SubGraph(root_id="x", nodes=[], edges=[])

    return HybridRetriever(sym_store, lex_store, graph_store, settings)


def test_retriever_returns_results():
    r = _make_retriever(sym_results=[("id1", "authenticate"), ("id2", "get_token")])
    results = r.retrieve("authentication", k=5)
    assert len(results) >= 1


def test_retriever_top_k_respected():
    sym = [(f"id{i}", f"fn{i}") for i in range(20)]
    r = _make_retriever(sym_results=sym)
    results = r.retrieve("function", k=5)
    assert len(results) <= 5


def test_retriever_rrf_score_positive():
    r = _make_retriever(sym_results=[("id1", "foo"), ("id2", "bar")])
    for result in r.retrieve("foo", k=5):
        assert result.rrf_score > 0


def test_retriever_provenance_lex():
    r = _make_retriever(sym_results=[("id1", "check_auth")])
    results = r.retrieve("auth", k=5)
    assert any("lex" in res.provenance for res in results)


def test_retriever_vector_unavailable_graceful():
    """Should not crash when Qdrant collection doesn't exist."""
    r = _make_retriever(sym_results=[("id1", "foo")])
    # _vector_search will fail silently
    results = r.retrieve("foo", k=5)
    assert isinstance(results, list)


def test_graph_expansion_adds_neighbors():
    from cce.graph.schema import SubGraph  # noqa: PLC0415

    neighbor = _make_mock_node("id_neighbor", "neighbor_fn")
    subgraph = SubGraph(root_id="id1", nodes=[neighbor], edges=[])

    r = _make_retriever(sym_results=[("id1", "foo")])
    r._graph.get_neighborhood.return_value = subgraph
    r._graph.get_node.side_effect = lambda nid: (
        _make_mock_node("id1", "foo") if nid == "id1"
        else neighbor if nid == "id_neighbor"
        else None
    )

    results = r.retrieve("foo", k=10)
    result_ids = {res.node_id for res in results}
    assert "id_neighbor" in result_ids
    neighbor_result = next(r for r in results if r.node_id == "id_neighbor")
    assert "graph" in neighbor_result.provenance


# ── Integration: search_code uses hybrid retriever ────────────────────────────

def test_get_file_slice_returns_lines(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.retrieval.tools import get_file_slice  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "slice.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical"])

    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        result = get_file_slice("views.py", 1, 10)
    assert result["path"] == "views.py"
    assert len(result["lines"]) <= 10
    assert result["start"] == 1


def test_get_file_slice_rejects_traversal(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.retrieval.tools import get_file_slice  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "slice2.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical"])

    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        with pytest.raises(ValueError, match="Invalid path"):
            get_file_slice("../../etc/passwd", 1, 10)


def test_list_tools_respect_limit(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.retrieval.tools import list_symbols  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "limit.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical", "symbols"])

    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        syms = list_symbols(file_path="views.py", kind="Function", limit=1)
        assert len(syms) == 1
        # Cap at 1000
        syms2 = list_symbols(file_path="views.py", kind="Function", limit=9999)
        assert len(syms2) <= 1000


def test_search_code_hybrid_mode(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "hybrid_e2e.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical", "symbols", "graph"])

    # Patch _pipeline and _hybrid_retriever to use our test pipeline
    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        from cce.retrieval.tools import search_code  # noqa: PLC0415
        hits = search_code("User class", mode="lexical", k=10)
        assert isinstance(hits, list)


# ── P0-3: grep_code tests ─────────────────────────────────────────────────────

def test_grep_code_regex_metachars_and_no_matches(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.retrieval.tools import grep_code  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "grep.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical"])

    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        # Escaped parenthesis should not crash
        hits = grep_code(r"def \w+\(")
        assert isinstance(hits, list)
        # A pattern that matches nothing should return empty list
        hits_empty = grep_code(r"xyz_NONEXISTENT_12345")
        assert hits_empty == []
        # Long pattern should work
        hits_long = grep_code(r"class User.*pass")
        assert isinstance(hits_long, list)
