"""Phase 6c — React extractor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cce.extractors.react_extractor import ReactExtractor
from cce.extractors.framework_detector import detect_frameworks
from cce.graph.schema import EdgeKind, FrameworkTag, NodeKind

FIXTURES = Path(__file__).parent / "fixtures" / "sample_react"


@pytest.fixture
def extractor():
    return ReactExtractor()


def _extract(extractor, fname: str):
    path = FIXTURES / fname
    source = path.read_text(encoding="utf-8", errors="ignore")
    return extractor.extract(path, fname, source)


# ── Framework detection ────────────────────────────────────────────────────────

def test_detect_react_framework():
    frameworks = detect_frameworks(FIXTURES)
    assert FrameworkTag.REACT in frameworks


def test_can_handle_tsx_with_routes(extractor):
    path = FIXTURES / "App.tsx"
    source = path.read_text(encoding="utf-8", errors="ignore")
    assert extractor.can_handle(path, source)


def test_can_handle_tsx_with_api_calls(extractor):
    path = FIXTURES / "hooks" / "useAuth.ts"
    source = path.read_text(encoding="utf-8", errors="ignore")
    assert extractor.can_handle(path, source)


# ── React Router routes ────────────────────────────────────────────────────────

def test_extracts_browser_router_routes(extractor):
    data = _extract(extractor, "App.tsx")
    routes = [n for n in data.nodes if n.kind == NodeKind.ROUTE]
    assert len(routes) >= 2


def test_route_paths_captured(extractor):
    data = _extract(extractor, "App.tsx")
    paths = {n.name for n in data.nodes if n.kind == NodeKind.ROUTE}
    assert "/" in paths or any("/users" in p for p in paths)


def test_route_renders_edge(extractor):
    data = _extract(extractor, "App.tsx")
    renders_edges = [e for e in data.raw_edges if e.kind == EdgeKind.RENDERS]
    assert len(renders_edges) >= 1


def test_route_component_in_meta(extractor):
    data = _extract(extractor, "App.tsx")
    components = [
        route.meta.get("component")
        for route in data.nodes
        if route.kind == NodeKind.ROUTE
    ]
    components = [c for c in components if c]
    # At least one route must render a PascalCase (user-defined) component;
    # HTML elements like `div` are allowed as route components but don't count.
    assert any(c[0].isalpha() and c[0].isupper() for c in components), (
        f"expected at least one PascalCase component in {components}"
    )


# ── CALLS_API cross-stack edges ────────────────────────────────────────────────

def test_extracts_api_call_edges_from_app(extractor):
    data = _extract(extractor, "App.tsx")
    api_edges = [e for e in data.raw_edges if e.kind == EdgeKind.REFERENCES]
    assert len(api_edges) >= 1
    dst_names = {e.dst_qualified_name for e in api_edges}
    assert any("/api" in d for d in dst_names)


def test_extracts_api_call_edges_from_hook(extractor):
    path = FIXTURES / "hooks" / "useAuth.ts"
    source = path.read_text(encoding="utf-8", errors="ignore")
    data = extractor.extract(path, "hooks/useAuth.ts", source)
    api_edges = [e for e in data.raw_edges if e.kind == EdgeKind.REFERENCES]
    assert len(api_edges) >= 1
    dst_names = {e.dst_qualified_name for e in api_edges}
    assert any("/api" in d for d in dst_names)


def test_api_edge_confidence(extractor):
    data = _extract(extractor, "App.tsx")
    for e in (e for e in data.raw_edges if e.kind == EdgeKind.REFERENCES):
        assert 0 < e.confidence <= 1.0


# ── React Router (framework tag) ───────────────────────────────────────────────

def test_route_framework_tag(extractor):
    data = _extract(extractor, "App.tsx")
    for n in (n for n in data.nodes if n.kind == NodeKind.ROUTE):
        assert n.framework_tag == FrameworkTag.REACT
