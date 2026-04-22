"""LangGraph agent invocation endpoint with SSE streaming."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cce.agents.graph import get_agent_graph

router = APIRouter()


class AgentQuery(BaseModel):
    query: str
    thread_id: str = "default"


class AgentResponse(BaseModel):
    answer: str
    citations: list[dict] = []
    reasoning_steps: list[str] = []


def _initial_state(query: str) -> dict:
    """Build the per-turn input: append the new user message, reset per-turn counters.

    ``messages`` uses the ``add_messages`` reducer so the new HumanMessage is
    appended to any prior conversation in the checkpointer. The other fields
    have no reducer and therefore replace the stale values from the previous
    turn — this is what prevents loop_count / reasoning_steps from leaking
    across requests that reuse the same ``thread_id``.
    """
    return {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "loop_count": 0,
        "reasoning_steps": [],
        "retrieved_context": [],
    }


@router.post("/query", response_model=AgentResponse)
async def query(req: AgentQuery) -> AgentResponse:
    """Invoke the LangGraph app synchronously and return the final state."""
    app = get_agent_graph()
    cfg = {"configurable": {"thread_id": req.thread_id}}
    final = await app.ainvoke(_initial_state(req.query), config=cfg)
    return AgentResponse(
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        reasoning_steps=final.get("reasoning_steps", []),
    )


@router.post("/stream")
async def stream(req: AgentQuery) -> EventSourceResponse:
    """Stream LangGraph events (node transitions + tokens) as SSE."""
    app = get_agent_graph()
    cfg = {"configurable": {"thread_id": req.thread_id}}

    async def event_source() -> AsyncIterator[dict]:
        async for event in app.astream_events(_initial_state(req.query), config=cfg, version="v2"):
            yield {"event": event.get("event", "message"), "data": json.dumps(event, default=str)}

    return EventSourceResponse(event_source())
