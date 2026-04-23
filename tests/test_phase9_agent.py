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
        mock_cfg.return_value.offline = False  # F37: explicit False so offline guard doesn't fire
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

    state = {"loop_count": 10, "messages": []}
    with patch("cce.agents.nodes.get_settings") as ms:
        ms.return_value.agent.max_retrieval_loops = 3
        assert should_continue(state) == "respond"


def test_should_continue_returns_plan_under_cap():
    from cce.agents.nodes import should_continue  # noqa: PLC0415

    state = {"loop_count": 1, "messages": []}
    with patch("cce.agents.nodes.get_settings") as ms:
        ms.return_value.agent.max_retrieval_loops = 5
        assert should_continue(state) == "plan"


def test_has_tool_calls_returns_retrieve_with_tool_calls():
    from cce.agents.nodes import has_tool_calls  # noqa: PLC0415
    from langchain_core.messages import AIMessage  # noqa: PLC0415

    msg = AIMessage(content="", tool_calls=[{"name": "search_code", "id": "1", "args": {}}])
    assert has_tool_calls({"messages": [msg]}) == "retrieve"


def test_has_tool_calls_returns_respond_without_tool_calls():
    from cce.agents.nodes import has_tool_calls  # noqa: PLC0415
    from langchain_core.messages import AIMessage  # noqa: PLC0415

    assert has_tool_calls({"messages": [AIMessage(content="done")]}) == "respond"
    assert has_tool_calls({"messages": []}) == "respond"


def test_responder_node_with_mocked_llm():
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: PLC0415
    from cce.agents.nodes import responder_node  # noqa: PLC0415

    state = {"messages": [HumanMessage(content="query")], "retrieved_context": [], "reasoning_steps": []}
    mock_resp = AIMessage(content="Here is the answer: authentication is in middleware.py")

    with patch("cce.agents.nodes.get_responder_llm") as mock_llm:
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


# ── P0-4 citation validation ──────────────────────────────────────────────────

def test_validate_citations_file_level():
    from cce.agents.nodes import _validate_citations  # noqa: PLC0415

    citations = [{"id": 1, "symbol": "x", "file": "x.py", "line_start": 0, "line_end": 0}]
    answer = "See x.py:42 for details."
    sanitised, dropped = _validate_citations(answer, citations)
    assert "x.py:42" in sanitised
    assert not dropped


def test_validate_citations_out_of_range():
    from cce.agents.nodes import _validate_citations  # noqa: PLC0415

    citations = [{"id": 1, "symbol": "x", "file": "x.py", "line_start": 10, "line_end": 20}]
    answer = "See x.py:5 and x.py:15 for details."
    sanitised, dropped = _validate_citations(answer, citations)
    assert "x.py:?" in sanitised
    assert "x.py:15" in sanitised
    assert "x.py:5" in dropped


def test_retriever_structured_tool_error_and_reasoner_counts_it():
    from langchain_core.messages import AIMessage, ToolMessage  # noqa: PLC0415
    from cce.agents.nodes import retriever_node, reasoner_node  # noqa: PLC0415

    # Mock a tool that always raises
    bad_tool = MagicMock()
    bad_tool.name = "bad_tool"
    bad_tool.invoke.side_effect = ValueError("something broke")

    with patch("cce.agents.tools.ALL_TOOLS", [bad_tool]):
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "bad_tool", "args": {}, "id": "tc-1"}]),
            ],
            "retrieved_context": [],
        }
        result = retriever_node(state)

    assert "messages" in result
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert '"status": "error"' in tool_msgs[0].content
    assert '"error_class": "ValueError"' in tool_msgs[0].content

    # Feed into reasoner
    state["messages"].extend(result["messages"])
    reasoner_result = reasoner_node(state)
    assert reasoner_result.get("tool_errors", 0) == 1
    assert any("error(s)" in s for s in reasoner_result["reasoning_steps"])


# ── P0-1: leaked tool-call parsing ────────────────────────────────────────────

def test_parse_leaked_tool_calls_shape_1():
    from cce.agents.nodes import _parse_leaked_tool_calls  # noqa: PLC0415

    content = 'to=functions.search_code {"query": "auth"}'
    calls = _parse_leaked_tool_calls(content, {"search_code"})
    assert len(calls) == 1
    assert calls[0]["name"] == "search_code"
    assert calls[0]["args"]["query"] == "auth"


def test_parse_leaked_tool_calls_shape_2():
    from cce.agents.nodes import _parse_leaked_tool_calls  # noqa: PLC0415

    content = '{"tool_uses":[{"recipient_name":"functions.search_code","parameters":{"query":"auth"}}]}'
    calls = _parse_leaked_tool_calls(content, {"search_code"})
    assert len(calls) == 1
    assert calls[0]["name"] == "search_code"


def test_parse_leaked_tool_calls_shape_3():
    from cce.agents.nodes import _parse_leaked_tool_calls  # noqa: PLC0415

    content = '{"name":"search_code","arguments":{"query":"auth"}}'
    calls = _parse_leaked_tool_calls(content, {"search_code"})
    assert len(calls) == 1
    assert calls[0]["name"] == "search_code"


def test_parse_leaked_tool_calls_empty():
    from cce.agents.nodes import _parse_leaked_tool_calls  # noqa: PLC0415

    assert _parse_leaked_tool_calls("", {"search_code"}) == []
    assert _parse_leaked_tool_calls("no leak here", {"search_code"}) == []


def test_parse_leaked_tool_calls_malformed():
    from cce.agents.nodes import _parse_leaked_tool_calls  # noqa: PLC0415

    content = 'to=functions.search_code {broken json'
    calls = _parse_leaked_tool_calls(content, {"search_code"})
    assert calls == []


# ── P0-4: more citation validation ────────────────────────────────────────────

def test_validate_citations_valid_range():
    from cce.agents.nodes import _validate_citations  # noqa: PLC0415

    citations = [{"id": 1, "symbol": "x", "file": "x.py", "line_start": 10, "line_end": 20}]
    answer = "See x.py:15 for details."
    sanitised, dropped = _validate_citations(answer, citations)
    assert "x.py:15" in sanitised
    assert not dropped


def test_validate_citations_windows_vs_unix_path():
    from cce.agents.nodes import _validate_citations  # noqa: PLC0415

    citations = [{"id": 1, "symbol": "x", "file": "src/x.py", "line_start": 5, "line_end": 5}]
    answer = "See x.py:5."
    sanitised, dropped = _validate_citations(answer, citations)
    assert "x.py:5" in sanitised
    assert not dropped


def test_validate_citations_multiple_in_sentence():
    from cce.agents.nodes import _validate_citations  # noqa: PLC0415

    citations = [
        {"id": 1, "symbol": "x", "file": "x.py", "line_start": 1, "line_end": 10},
        {"id": 2, "symbol": "y", "file": "y.py", "line_start": 5, "line_end": 5},
    ]
    answer = "See x.py:3 and y.py:5."
    sanitised, dropped = _validate_citations(answer, citations)
    assert "x.py:3" in sanitised
    assert "y.py:5" in sanitised
    assert not dropped
