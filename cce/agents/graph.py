"""LangGraph StateGraph assembly.

Wires nodes from `cce.agents.nodes` into a directed graph with a reasoning
loop, compiled with a checkpointer for persistence.
"""

from __future__ import annotations

import asyncio
import threading
from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cce.agents.nodes import (
    has_tool_calls,
    planner_node,
    reasoner_node,
    responder_node,
    retriever_node,
    should_continue,
)
from cce.agents.state import AgentState
from cce.config import get_settings

# Dedicated background event loop used to construct an aiosqlite.Connection
# synchronously. The connection itself is loop-agnostic once created, so the
# AsyncSqliteSaver built on top of it can be driven from any event loop
# (FastAPI request loop, pytest-asyncio loop, …).
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None


def _ensure_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    if _bg_loop is None:
        _bg_loop = asyncio.new_event_loop()
        _bg_thread = threading.Thread(
            target=_bg_loop.run_forever, name="cce-aiosqlite-loop", daemon=True,
        )
        _bg_thread.start()
    return _bg_loop


def _build_checkpointer():
    settings = get_settings()
    if settings.agent.checkpointer == "sqlite":
        try:
            import aiosqlite  # noqa: PLC0415
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415
        except ImportError:
            # aiosqlite not installed → fall back to MemorySaver so the graph
            # still works in async contexts (HTTP server, pytest-asyncio).
            return MemorySaver()

        loop = _ensure_bg_loop()

        async def _build() -> "AsyncSqliteSaver":
            # AsyncSqliteSaver.__init__ calls asyncio.get_running_loop(), so we
            # have to construct both conn AND saver on the bg loop.
            conn = await aiosqlite.connect(str(settings.paths.agent_checkpoint))
            return AsyncSqliteSaver(conn)

        fut = asyncio.run_coroutine_threadsafe(_build(), loop)
        return fut.result(timeout=10)
    return MemorySaver()


@lru_cache(maxsize=1)
def get_agent_graph():
    """Build and cache the compiled LangGraph app."""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("reasoner", reasoner_node)
    builder.add_node("responder", responder_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        has_tool_calls,
        {"retrieve": "retriever", "respond": "responder"},
    )
    builder.add_edge("retriever", "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        should_continue,
        {"plan": "planner", "respond": "responder"},
    )
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=_build_checkpointer())
