"""Phase 9 — Agent, LLM factory, and MCP server tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── LLM factory ───────────────────────────────────────────────────────────────

def test_get_llm_raises_without_provider():
    from cce.agents.llm import get_llm  # noqa: PLC0415
    get_llm.cache_clear()
    with patch("cce.agents.llm.get_settings") as mock_cfg:
        mock_cfg.return_value.agent.llm_provider = "invalid_xyz"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm()
    get_llm.cache_clear()


def test_system_prompt_contains_tool_guidance():
    from cce.agents.llm import SYSTEM_PROMPT  # noqa: PLC0415
    assert "tools" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower() or "symbol" in SYSTEM_PROMPT.lower()


# ── Agent nodes ───────────────────────────────────────────────────────────────

def test_planner_falls_back_without_llm():
    """planner_node uses _fallback_search when LLM fails."""
    from cce.agents.nodes import planner_node  # noqa: PLC0415

    state = {"query": "where is authentication handled?", "messages": [], "loop_count": 0}
    with patch("cce.agents.nodes.get_llm", side_effect=Exception("no LLM")):
        result = planner_node(state)
    assert "messages" in result
    assert result["loop_count"] == 1


def test_retriever_handles_no_tool_calls():
    """retriever_node is a no-op when last message has no tool_calls."""
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    from cce.agents.nodes import retriever_node  # noqa: PLC0415

    state = {"messages": [HumanMessage(content="hello")], "retrieved_context": []}
    result = retriever_node(state)
    assert result == {}


def test_should_continue_returns_respond_at_max_loops():
    from cce.agents.nodes import should_continue  # noqa: PLC0415
    from langchain_core.messages import AIMessage  # noqa: PLC0415

    state = {
        "loop_count": 10,
        "messages": [AIMessage(content="done", tool_calls=[{"name": "search_code", "id": "x", "args": {}}])],
    }
    with patch("cce.agents.nodes.get_settings") as ms:
        ms.return_value.agent.max_retrieval_loops = 3
        assert should_continue(state) == "respond"


def test_should_continue_returns_retrieve_with_tool_calls():
    from cce.agents.nodes import should_continue  # noqa: PLC0415
    from langchain_core.messages import AIMessage  # noqa: PLC0415

    msg = AIMessage(content="", tool_calls=[{"name": "search_code", "id": "1", "args": {}}])
    state = {"loop_count": 1, "messages": [msg]}
    with patch("cce.agents.nodes.get_settings") as ms:
        ms.return_value.agent.max_retrieval_loops = 5
        assert should_continue(state) == "retrieve"


def test_responder_node_with_mocked_llm():
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: PLC0415
    from cce.agents.nodes import responder_node  # noqa: PLC0415

    state = {"messages": [HumanMessage(content="query")], "retrieved_context": [], "reasoning_steps": []}
    mock_resp = AIMessage(content="Here is the answer: authentication is in middleware.py")

    with patch("cce.agents.nodes.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = mock_resp
        result = responder_node(state)

    assert "answer" in result
    assert "authentication" in result["answer"]


# ── Agent graph compiles + runs end-to-end ────────────────────────────────────

def test_agent_graph_compiles():
    from cce.agents.graph import get_agent_graph  # noqa: PLC0415
    app = get_agent_graph()
    assert app is not None


@pytest.mark.asyncio
async def test_agent_ainvoke_with_mock_llm():
    from langchain_core.messages import AIMessage  # noqa: PLC0415
    from cce.agents.graph import get_agent_graph  # noqa: PLC0415

    get_agent_graph.cache_clear()
    stub_resp = AIMessage(content="Auth is handled in middleware")
    with patch("cce.agents.nodes.get_llm") as mock_llm:
        mock_llm.return_value.bind_tools.return_value.invoke.return_value = stub_resp
        mock_llm.return_value.invoke.return_value = stub_resp
        app = get_agent_graph()
        result = await app.ainvoke(
            {"query": "how is auth handled?", "messages": []},
            config={"configurable": {"thread_id": "test-thread"}},
        )
    assert "answer" in result


# ── MCP server ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from cce.server.app import create_app  # noqa: PLC0415
    return TestClient(create_app())


def test_mcp_initialize(client):
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["protocolVersion"] is not None
    assert body["result"]["serverInfo"]["name"] == "code-context-engine"


def test_mcp_tools_list(client):
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert resp.status_code == 200
    tools = resp.json()["result"]["tools"]
    assert len(tools) >= 7
    names = {t["name"] for t in tools}
    assert "search_code" in names
    assert "get_symbol" in names
    assert "get_neighborhood" in names


def test_mcp_get_tools_endpoint(client):
    resp = client.get("/mcp/tools")
    assert resp.status_code == 200
    assert "tools" in resp.json()


def test_mcp_unknown_method_returns_error(client):
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "unknown/method"})
    assert resp.status_code == 200
    assert resp.json()["error"]["code"] == -32601


def test_mcp_tools_call_search_code(client):
    with patch("cce.retrieval.tools.search_code", return_value=[]):
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "search_code", "arguments": {"query": "auth"}},
        })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
