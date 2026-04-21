"""LangGraph StateGraph assembly.

Wires nodes from `cce.agents.nodes` into a directed graph with a reasoning
loop, compiled with a checkpointer for persistence.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cce.agents.nodes import (
    planner_node,
    reasoner_node,
    responder_node,
    retriever_node,
    should_continue,
)
from cce.agents.state import AgentState
from cce.config import get_settings


def _build_checkpointer():
    settings = get_settings()
    if settings.agent.checkpointer == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(str(settings.paths.agent_checkpoint))
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
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        should_continue,
        {"plan": "planner", "respond": "responder"},
    )
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=_build_checkpointer())
