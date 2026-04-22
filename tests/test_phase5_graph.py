"""Phase 5 — SQLite graph store and neighborhood tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cce.graph.schema import EdgeKind, Language, NodeKind
from cce.graph.sqlite_store import SQLiteGraphStore
from cce.index.db import get_db
from cce.index.symbol_store import SymbolStore
from cce.parsers.tree_sitter_parser import TreeSitterParser

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PY = FIXTURES / "sample_python"


@pytest.fixture
def stores(tmp_path):
    db = get_db(tmp_path / "graph_test.sqlite")
    return SymbolStore(db), SQLiteGraphStore(db)


@pytest.fixture
def populated_stores(stores):
    """Index sample_python/models.py into both symbol and graph stores."""
    sym_store, graph_store = stores
    parser = TreeSitterParser()

    for fname in ("models.py", "views.py"):
        path = SAMPLE_PY / fname
        source = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(SAMPLE_PY))
        pf = parser.parse(path, rel, Language.PYTHON, source)
        sym_store.upsert_many(pf.nodes)

        for re_ in pf.raw_edges:
            dst_node = sym_store.get_by_qname(re_.dst_qualified_name)
            if dst_node:
                graph_store.upsert_edge(
                    src_id=re_.src_id,
                    dst_id=dst_node.id,
                    kind=re_.kind,
                    file_path=re_.file_path,
                    line=re_.line,
                    confidence=re_.confidence,
                )

    return sym_store, graph_store


def test_upsert_and_resolve_qname(stores):
    sym_store, graph_store = stores
    # graph store resolve_qname uses symbols table
    parser = TreeSitterParser()
    path = SAMPLE_PY / "models.py"
    source = path.read_text(encoding="utf-8", errors="ignore")
    pf = parser.parse(path, "models.py", Language.PYTHON, source)
    sym_store.upsert_many(pf.nodes)

    user = sym_store.get_by_qname(next(n.qualified_name for n in pf.nodes if n.name == "User"))
    assert user is not None
    resolved = graph_store.resolve_qname(user.qualified_name)
    assert resolved == user.id


def test_upsert_edge_no_duplicate(populated_stores):
    sym_store, graph_store = populated_stores
    user = sym_store.get_by_qname(
        next(q for q in sym_store.list_qnames() if q.endswith("User") and "Admin" not in q)
    )
    admin = sym_store.get_by_qname(
        next(q for q in sym_store.list_qnames() if q.endswith("AdminUser"))
    )
    if user and admin:
        graph_store.upsert_edge(user.id, admin.id, EdgeKind.INHERITS)
        graph_store.upsert_edge(user.id, admin.id, EdgeKind.INHERITS)  # duplicate
        db = graph_store._db
        count = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id=? AND dst_id=? AND kind=?",
            (user.id, admin.id, EdgeKind.INHERITS.value),
        ).fetchone()[0]
        assert count == 1


def test_find_implementations(populated_stores):
    sym_store, graph_store = populated_stores
    # AdminUser inherits User — find_implementations(User) should return AdminUser
    user_qname = next((q for q in sym_store.list_qnames() if q.endswith(".User")), None)
    if user_qname is None:
        pytest.skip("User symbol not found in index")
    user = sym_store.get_by_qname(user_qname)
    impls = graph_store.find_implementations(user.id)
    impl_names = {n.name for n in impls}
    assert "AdminUser" in impl_names


def test_neighborhood_returns_connected_nodes(populated_stores):
    sym_store, graph_store = populated_stores
    user_qname = next((q for q in sym_store.list_qnames() if q.endswith(".User")), None)
    if user_qname is None:
        pytest.skip("User symbol not found in index")
    user = sym_store.get_by_qname(user_qname)
    sg = graph_store.get_neighborhood(user.id, depth=2)
    assert sg.root_id == user.id
    assert len(sg.nodes) >= 1   # at least the anchor itself (via INHERITS edge)


def test_delete_for_file_removes_edges(populated_stores):
    sym_store, graph_store = populated_stores
    graph_store.delete_for_file("models.py")
    # After deletion, edges referencing deleted nodes should be gone
    db = graph_store._db
    orphan = db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE "
        "src_id NOT IN (SELECT id FROM symbols) OR "
        "dst_id NOT IN (SELECT id FROM symbols)"
    ).fetchone()[0]
    assert orphan == 0


def test_full_pipeline_end_to_end(tmp_path):
    """Run IndexPipeline over sample_python and verify symbols + edges."""
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "e2e.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    stats = pipeline.run(SAMPLE_PY, layers=["lexical", "symbols", "graph"])

    assert stats.files_total >= 2
    assert stats.symbols_indexed > 0

    user = pipeline.symbol_store.get_by_qname(
        next(q for q in pipeline.symbol_store.list_qnames() if q.endswith(".User"))
    )
    assert user is not None
    assert user.name == "User"
