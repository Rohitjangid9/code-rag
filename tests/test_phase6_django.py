"""Phase 6a — Django extractor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cce.extractors.django_extractor import DjangoExtractor
from cce.extractors.framework_detector import detect_frameworks, file_belongs_to
from cce.graph.schema import EdgeKind, FrameworkTag, NodeKind

FIXTURES = Path(__file__).parent / "fixtures" / "sample_django"


@pytest.fixture
def extractor():
    return DjangoExtractor()


def _extract(extractor, fname: str):
    path = FIXTURES / fname
    source = path.read_text(encoding="utf-8", errors="ignore")
    return extractor.extract(path, fname, source)


# ── Framework detection ────────────────────────────────────────────────────────

def test_detect_django_framework():
    frameworks = detect_frameworks(FIXTURES)
    assert FrameworkTag.DJANGO in frameworks or FrameworkTag.DRF in frameworks


def test_file_belongs_to_models(extractor):
    path = FIXTURES / "models.py"
    source = path.read_text(encoding="utf-8", errors="ignore")
    relevant = file_belongs_to(path, source, {FrameworkTag.DJANGO, FrameworkTag.DRF})
    assert FrameworkTag.DJANGO in relevant


# ── Models ────────────────────────────────────────────────────────────────────

def test_extracts_model_nodes(extractor):
    data = _extract(extractor, "models.py")
    model_names = {n.name for n in data.nodes if n.kind == NodeKind.MODEL}
    assert "Category" in model_names
    assert "Article" in model_names


def test_model_framework_tag(extractor):
    data = _extract(extractor, "models.py")
    for n in data.nodes:
        if n.kind == NodeKind.MODEL:
            assert n.framework_tag == FrameworkTag.DJANGO


def test_model_fields_in_meta(extractor):
    data = _extract(extractor, "models.py")
    article = next(n for n in data.nodes if n.name == "Article" and n.kind == NodeKind.MODEL)
    field_names = {f["name"] for f in article.meta.get("fields", [])}
    assert "title" in field_names
    assert "body" in field_names
    assert "published" in field_names


# ── DRF Serializers ────────────────────────────────────────────────────────────

def test_extracts_serializer_nodes(extractor):
    data = _extract(extractor, "serializers.py")
    serializer_names = {n.name for n in data.nodes if n.kind == NodeKind.SERIALIZER}
    assert "CategorySerializer" in serializer_names
    assert "ArticleSerializer" in serializer_names


def test_serializer_uses_model_edge(extractor):
    data = _extract(extractor, "serializers.py")
    uses_model_edges = [e for e in data.raw_edges if e.kind == EdgeKind.USES_MODEL]
    dst_names = {e.dst_qualified_name for e in uses_model_edges}
    assert "Category" in dst_names
    assert "Article" in dst_names


def test_serializer_framework_tag(extractor):
    data = _extract(extractor, "serializers.py")
    for n in data.nodes:
        if n.kind == NodeKind.SERIALIZER:
            assert n.framework_tag == FrameworkTag.DRF


# ── Signals ────────────────────────────────────────────────────────────────────

def test_extracts_signal_edges(extractor):
    data = _extract(extractor, "views.py")
    signal_edges = [e for e in data.raw_edges if e.kind == EdgeKind.HANDLES_SIGNAL]
    assert len(signal_edges) >= 1
    # @receiver(post_save, sender=Article) on_article_saved
    dst_names = {e.dst_qualified_name for e in signal_edges}
    assert any("on_article_saved" in d for d in dst_names)


# ── URL patterns ───────────────────────────────────────────────────────────────

def test_extracts_url_pattern_nodes(extractor):
    data = _extract(extractor, "urls.py")
    url_nodes = [n for n in data.nodes if n.kind == NodeKind.URL_PATTERN]
    # Should find at least the api/v1/ and api/v1/health/ patterns
    assert len(url_nodes) >= 1


def test_url_routes_to_edge(extractor):
    data = _extract(extractor, "urls.py")
    routes_edges = [e for e in data.raw_edges if e.kind == EdgeKind.ROUTES_TO]
    assert len(routes_edges) >= 1


# ── End-to-end pipeline ────────────────────────────────────────────────────────

def test_pipeline_indexes_django_framework(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "django_e2e.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    stats = pipeline.run(FIXTURES, layers=["lexical", "symbols", "graph", "framework"])

    assert stats.symbols_indexed > 0
    # Should have indexed Model and Serializer nodes
    conn = pipeline.symbol_store._db.conn
    model_count = conn.execute("SELECT COUNT(*) FROM symbols WHERE kind='Model'").fetchone()[0]
    serializer_count = conn.execute("SELECT COUNT(*) FROM symbols WHERE kind='Serializer'").fetchone()[0]
    assert model_count >= 2
    assert serializer_count >= 2
