"""Phase 6b — FastAPI extractor tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cce.extractors.fastapi_extractor import FastAPIExtractor
from cce.extractors.framework_detector import detect_frameworks
from cce.graph.schema import EdgeKind, FrameworkTag, NodeKind

FIXTURES = Path(__file__).parent / "fixtures" / "sample_fastapi"


@pytest.fixture
def extractor():
    return FastAPIExtractor()


def _extract(extractor, fname: str):
    path = FIXTURES / fname
    source = path.read_text(encoding="utf-8", errors="ignore")
    return extractor.extract(path, fname, source)


# ── Framework detection ────────────────────────────────────────────────────────

def test_detect_fastapi_framework():
    frameworks = detect_frameworks(FIXTURES)
    assert FrameworkTag.FASTAPI in frameworks


def test_can_handle_main_py(extractor):
    path = FIXTURES / "main.py"
    source = path.read_text(encoding="utf-8", errors="ignore")
    assert extractor.can_handle(path, source)


def test_can_handle_models_py(extractor):
    path = FIXTURES / "models.py"
    source = path.read_text(encoding="utf-8", errors="ignore")
    # models.py contains Pydantic BaseModel classes that the FastAPI extractor handles
    assert extractor.can_handle(path, source)


# ── Route extraction ───────────────────────────────────────────────────────────

def test_extracts_route_nodes(extractor):
    data = _extract(extractor, "main.py")
    routes = [n for n in data.nodes if n.kind == NodeKind.ROUTE]
    assert len(routes) >= 3  # /health, users, articles routes


def test_route_has_method(extractor):
    data = _extract(extractor, "main.py")
    for route in (n for n in data.nodes if n.kind == NodeKind.ROUTE):
        assert len(route.meta.get("methods", [])) >= 1


def test_route_has_handler_qname(extractor):
    data = _extract(extractor, "main.py")
    for route in (n for n in data.nodes if n.kind == NodeKind.ROUTE):
        assert route.meta.get("handler"), f"Route {route.name} has no handler"


def test_routes_to_edge_emitted(extractor):
    data = _extract(extractor, "main.py")
    routes_edges = [e for e in data.raw_edges if e.kind == EdgeKind.ROUTES_TO]
    assert len(routes_edges) >= 3


def test_health_route_detected(extractor):
    data = _extract(extractor, "main.py")
    routes = [n for n in data.nodes if n.kind == NodeKind.ROUTE]
    route_paths = [n.name for n in routes]
    assert any("/health" in p for p in route_paths)


def test_response_model_captured(extractor):
    data = _extract(extractor, "main.py")
    routes = [n for n in data.nodes if n.kind == NodeKind.ROUTE and n.meta.get("response_model")]
    assert len(routes) >= 1
    models = {r.meta["response_model"] for r in routes}
    assert any("Response" in m for m in models)


# ── Pydantic models ────────────────────────────────────────────────────────────

def test_extracts_pydantic_models(extractor):
    data = _extract(extractor, "models.py")
    pm_names = {n.name for n in data.nodes if n.kind == NodeKind.PYDANTIC_MODEL}
    assert "UserCreate" in pm_names
    assert "UserResponse" in pm_names
    assert "ArticleCreate" in pm_names


def test_pydantic_framework_tag(extractor):
    data = _extract(extractor, "models.py")
    for n in data.nodes:
        if n.kind == NodeKind.PYDANTIC_MODEL:
            assert n.framework_tag == FrameworkTag.FASTAPI


# ── Depends ────────────────────────────────────────────────────────────────────

def test_extracts_depends_edges(extractor):
    data = _extract(extractor, "main.py")
    depends_edges = [e for e in data.raw_edges if e.kind == EdgeKind.DEPENDS_ON]
    assert len(depends_edges) >= 1
    dst_names = {e.dst_qualified_name for e in depends_edges}
    assert "get_db" in dst_names or any("get_db" in d for d in dst_names)


# ── include_router ─────────────────────────────────────────────────────────────

def test_extracts_mounts_router_edges(extractor):
    data = _extract(extractor, "main.py")
    mount_edges = [e for e in data.raw_edges if e.kind == EdgeKind.MOUNTS_ROUTER]
    assert len(mount_edges) >= 2  # two app.include_router() calls


# ── End-to-end pipeline ────────────────────────────────────────────────────────

def test_pipeline_indexes_fastapi_framework(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "fastapi_e2e.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    stats = pipeline.run(FIXTURES, layers=["lexical", "symbols", "graph", "framework"])

    conn = pipeline.symbol_store._db.conn
    route_count = conn.execute("SELECT COUNT(*) FROM symbols WHERE kind='Route'").fetchone()[0]
    pm_count = conn.execute("SELECT COUNT(*) FROM symbols WHERE kind='PydanticModel'").fetchone()[0]
    assert route_count >= 3
    assert pm_count >= 2


def test_cross_file_include_router_prefix(tmp_path):
    """F4: Routes in one file mounted with a prefix from another file get effective_path."""
    import json as _json  # noqa: PLC0415
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text(
        'from fastapi import FastAPI\n'
        'from app.routes import router\n\n'
        'app = FastAPI()\n'
        'app.include_router(router, prefix="/agent")\n'
    )
    (app_dir / "routes.py").write_text(
        'from fastapi import APIRouter\n\n'
        'router = APIRouter()\n\n'
        '@router.post("/query")\n'
        'def query():\n'
        '    pass\n'
    )

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "cross.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(tmp_path, layers=["lexical", "symbols", "graph", "framework"])

    conn = pipeline.symbol_store._db.conn
    row = conn.execute(
        "SELECT meta FROM symbols WHERE kind='Route' AND json_extract(meta,'$.local_path') = '/query'"
    ).fetchone()
    assert row is not None
    meta = _json.loads(row["meta"])
    assert meta.get("effective_path") == "/agent/query"

    # F5: find_route_handler should resolve the cross-file route
    from cce.retrieval.tools import find_route_handler  # noqa: PLC0415
    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        info = find_route_handler("POST", "/agent/query")
    assert info.handler_qname.endswith(".query")


def test_list_routes_returns_routes(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.retrieval.tools import list_routes  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "routes.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical", "symbols", "graph", "framework"])

    with patch("cce.retrieval.tools._pipeline", return_value=pipeline):
        routes = list_routes()
    assert len(routes) >= 3
    # Each route should have an effective_path (same as path when no cross-file prefix)
    for r in routes:
        assert r.effective_path is not None
