"""LangGraph agent invocation endpoint with SSE streaming.

F32: A per-``thread_id`` ``asyncio.Lock`` prevents two simultaneous requests
from running the same conversation concurrently, which would corrupt the
checkpointer state and produce interleaved traces.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cce.agents.graph import get_agent_graph
from cce.telemetry import get_tracer

router = APIRouter()

# F32: per-thread_id concurrency guard
_THREAD_LOCKS: dict[str, asyncio.Lock] = {}
_THREAD_LOCKS_META: dict[str, int] = {}  # reference counts for cleanup


def _get_thread_lock(thread_id: str) -> asyncio.Lock:
    """Return the asyncio.Lock for *thread_id*, creating it if absent."""
    if thread_id not in _THREAD_LOCKS:
        _THREAD_LOCKS[thread_id] = asyncio.Lock()
        _THREAD_LOCKS_META[thread_id] = 0
    return _THREAD_LOCKS[thread_id]


def _release_thread_lock(thread_id: str) -> None:
    """Decrement ref-count and evict the lock when the thread is idle."""
    _THREAD_LOCKS_META[thread_id] = max(0, _THREAD_LOCKS_META.get(thread_id, 1) - 1)
    if _THREAD_LOCKS_META[thread_id] == 0 and not _THREAD_LOCKS[thread_id].locked():
        _THREAD_LOCKS.pop(thread_id, None)
        _THREAD_LOCKS_META.pop(thread_id, None)


class AgentQuery(BaseModel):
    query: str
    thread_id: str = "default"
    # F-M7: optional override of the repo this query targets.  Falls back to
    # the server-wide CCE_REPO_ROOT / walk-up detection when None.
    repo_root: str | None = None


class AgentResponse(BaseModel):
    answer: str
    citations: list[dict] = []
    reasoning_steps: list[str] = []


@contextmanager
def _bind_repo_root(repo_root: str | None):
    """Temporarily set CCE_REPO_ROOT for the duration of one request.

    Restoring the previous value afterwards keeps the server safe for
    concurrent requests targeting different repos.  The agent graph and
    retrieval caches are repo-keyed (F-M2) so switching env vars inside
    one async task does not leak into another.
    """
    if not repo_root:
        yield
        return
    resolved = str(Path(repo_root).resolve())
    previous = os.environ.get("CCE_REPO_ROOT")
    os.environ["CCE_REPO_ROOT"] = resolved
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CCE_REPO_ROOT", None)
        else:
            os.environ["CCE_REPO_ROOT"] = previous


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
    """Invoke the LangGraph app synchronously and return the final state.

    F32: serialises concurrent requests for the same thread_id.
    """
    lock = _get_thread_lock(req.thread_id)
    _THREAD_LOCKS_META[req.thread_id] = _THREAD_LOCKS_META.get(req.thread_id, 0) + 1
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail=f"A request for thread_id={req.thread_id!r} is already in progress.",
        )
    async with lock:
        with _bind_repo_root(req.repo_root), get_tracer().start_as_current_span(
            "cce.http.agent.query"
        ) as span:  # F33
            span.set_attribute("thread_id", req.thread_id)
            span.set_attribute("query", req.query[:200])
            if req.repo_root:
                span.set_attribute("repo_root", req.repo_root)
            try:
                app = get_agent_graph()
                cfg = {"configurable": {"thread_id": req.thread_id}}
                final = await app.ainvoke(_initial_state(req.query), config=cfg)
                return AgentResponse(
                    answer=final.get("answer", ""),
                    citations=final.get("citations", []),
                    reasoning_steps=final.get("reasoning_steps", []),
                )
            except Exception as exc:
                span.record_exception(exc)
                raise
            finally:
                _release_thread_lock(req.thread_id)


@router.post("/stream")
async def stream(req: AgentQuery) -> EventSourceResponse:
    """Stream LangGraph events (node transitions + tokens) as SSE.

    F32: serialises concurrent streams for the same thread_id.
    """
    lock = _get_thread_lock(req.thread_id)
    _THREAD_LOCKS_META[req.thread_id] = _THREAD_LOCKS_META.get(req.thread_id, 0) + 1
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail=f"A stream for thread_id={req.thread_id!r} is already in progress.",
        )

    async def event_source() -> AsyncIterator[dict]:
        async with lock:
            try:
                with _bind_repo_root(req.repo_root):
                    app = get_agent_graph()
                    cfg = {"configurable": {"thread_id": req.thread_id}}
                    async for event in app.astream_events(
                        _initial_state(req.query), config=cfg, version="v2"
                    ):
                        yield {"event": event.get("event", "message"),
                               "data": json.dumps(event, default=str)}
            finally:
                _release_thread_lock(req.thread_id)

    return EventSourceResponse(event_source())
