"""Phase 7 — Chunker, embedder interface, vector store, and semantic search tests.

The NomicEmbedder requires GPU + model download, so we mock it throughout.
Real integration tests are gated behind the 'gpu' pytest mark.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cce.embeddings.chunker import Chunk, build_header, chunk_node, chunk_nodes
from cce.graph.schema import Language, NodeKind
from cce.graph.schema import Node

FIXTURES = Path(__file__).parent / "fixtures" / "sample_python"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_node(name="foo", kind=NodeKind.FUNCTION, line_start=1, line_end=5,
               file_path="app/views.py", fw=None) -> Node:
    from cce.parsers.tree_sitter_parser import _node_id_from_qname  # noqa: PLC0415

    qname = f"app.views.{name}"
    return Node(
        id=_node_id_from_qname(qname),
        kind=kind,
        qualified_name=qname,
        name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        language=Language.PYTHON,
        framework_tag=fw,
        docstring="Does something useful.",
    )


# ── Chunker ────────────────────────────────────────────────────────────────────

def test_build_header_contains_path():
    node = _make_node()
    header = build_header(node)
    assert "app/views.py" in header
    assert "app.views.foo" in header
    assert "Function" in header
    assert "python" in header


def test_build_header_includes_docstring():
    node = _make_node()
    header = build_header(node)
    assert "Does something useful" in header


def test_chunk_node_extracts_body():
    node = _make_node(line_start=2, line_end=4)
    lines = ["# preamble", "def foo():", "    return 42", "    # done", "# end"]
    chunk = chunk_node(node, lines)
    assert "def foo" in chunk.body
    assert chunk.node_id == node.id
    assert chunk.qualified_name == node.qualified_name


def test_chunk_node_token_count_positive():
    node = _make_node(line_start=1, line_end=3)
    lines = ["def foo():", "    x = 1", "    return x"]
    chunk = chunk_node(node, lines)
    assert chunk.token_count > 0


def test_chunk_nodes_skips_missing_files():
    nodes = [_make_node(file_path="nonexistent.py")]
    chunks = chunk_nodes(nodes, {})
    assert chunks == []


def test_chunk_nodes_skips_tiny_symbols():
    node = _make_node(line_start=1, line_end=1)  # 0-line span
    lines = ["x = 1"]
    chunks = chunk_nodes([node], {"app/views.py": lines})
    assert chunks == []


def test_chunk_nodes_returns_chunks_for_valid_nodes():
    nodes = [_make_node(line_start=1, line_end=4)]
    lines = ["def foo():", "    x = 1", "    y = 2", "    return x + y"]
    chunks = chunk_nodes(nodes, {"app/views.py": lines})
    assert len(chunks) == 1
    assert chunks[0].path == "app/views.py"


def test_chunk_oversized_body_is_truncated():
    node = _make_node(line_start=1, line_end=100)
    long_line = "x = " + " ".join(["token"] * 20)
    lines = [long_line] * 100
    chunk = chunk_node(node, lines)
    assert chunk.token_count <= 2_000  # header + body budget


# ── Embedder interface ─────────────────────────────────────────────────────────

def test_embedder_protocol_respected():
    """A mock embedder that satisfies the Embedder ABC."""
    from cce.embeddings.embedder import Embedder  # noqa: PLC0415

    class FakeEmbedder(Embedder):
        backend_name = "fake"
        dim = 64

        def embed_documents(self, texts):
            return [[0.1] * 64 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 64

    emb = FakeEmbedder()
    vecs = emb.embed_documents(["def foo(): pass", "class Bar: pass"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 64

    q_vec = emb.embed_query("where is authentication handled?")
    assert len(q_vec) == 64


# ── Vector store (mocked Qdrant) ───────────────────────────────────────────────

def test_vector_store_collection_name_is_deterministic(tmp_path):
    from cce.config import Settings  # noqa: PLC0415

    with patch("qdrant_client.QdrantClient"):
        from cce.index.vector_store import VectorStore  # noqa: PLC0415

        cfg = Settings()
        cfg.paths.qdrant_path = tmp_path / "qdrant"

        store = VectorStore.__new__(VectorStore)
        store._dim = 3584

        root = Path("/some/repo")
        name1 = store.collection_name(root)
        name2 = store.collection_name(root)
        assert name1 == name2
        assert name1.startswith("cce_")


def test_vector_store_upsert_calls_qdrant(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.embeddings.chunker import Chunk  # noqa: PLC0415

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        from cce.index.vector_store import VectorStore  # noqa: PLC0415

        cfg = Settings()
        cfg.paths.qdrant_path = tmp_path / "qdrant"
        store = VectorStore(cfg)
        store.ensure_collection("cce_test")

        chunk = Chunk(
            node_id="abc", path="views.py", qualified_name="views.foo",
            kind="Function", header="# path: views.py", body="def foo(): pass",
        )
        store.upsert("cce_test", [(chunk, [0.1] * 3584)])

    mock_client.upsert.assert_called_once()


# ── Semantic search integration (mocked) ──────────────────────────────────────

def test_search_code_semantic_with_mock(tmp_path):
    """search_code(mode='semantic') calls the vector store when it's available."""
    from cce.config import Settings  # noqa: PLC0415

    mock_result = MagicMock()
    mock_result.score = 0.95
    mock_result.payload = {
        "node_id": "sym_abc",
        "path": "app/views.py",
        "qualified_name": "app.views.get_user",
        "header": "# path: app/views.py",
    }

    with (
        patch("cce.retrieval.tools._semantic_search", return_value=[]) as mock_sem,
        patch("cce.retrieval.tools._pipeline") as mock_pipe,
    ):
        mock_pipe.return_value.lexical_store.search.return_value = []
        mock_pipe.return_value.symbol_store.search.return_value = []

        from cce.retrieval.tools import search_code  # noqa: PLC0415

        hits = search_code("how is auth handled?", mode="semantic", k=5)
        # semantic mode calls _semantic_search
        mock_sem.assert_called_once()
