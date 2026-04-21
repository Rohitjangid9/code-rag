"""Smoke tests verifying the scaffolding wires together."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_openapi_includes_all_routers(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"].keys())
    assert "/health" in paths
    assert "/index" in paths
    assert "/search" in paths
    assert "/agent/query" in paths


def test_config_loads() -> None:
    from cce.config import get_settings

    s = get_settings()
    assert s.embedder.backend == "nomic"
    assert s.embedder.dim == 3584
    assert s.agent.checkpointer in {"memory", "sqlite"}


def test_agent_graph_compiles() -> None:
    from cce.agents.graph import get_agent_graph

    app = get_agent_graph()
    assert app is not None


def test_retrieval_tools_signatures_exist() -> None:
    from cce.retrieval import tools

    # Signatures must remain stable even when bodies are NotImplementedError.
    assert callable(tools.search_code)
    assert callable(tools.find_callers)
    assert callable(tools.get_route)
