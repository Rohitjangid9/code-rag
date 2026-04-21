"""LangGraph agent invocation endpoint with SSE streaming."""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter
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


@router.post("/query", response_model=AgentResponse)
async def query(req: AgentQuery) -> AgentResponse:
    """Invoke the LangGraph app synchronously and return the final state."""
    app = get_agent_graph()
    cfg = {"configurable": {"thread_id": req.thread_id}}
    final = await app.ainvoke({"query": req.query}, config=cfg)
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
        async for event in app.astream_events({"query": req.query}, config=cfg, version="v2"):
            yield {"event": event.get("event", "message"), "data": json.dumps(event, default=str)}

    return EventSourceResponse(event_source())
