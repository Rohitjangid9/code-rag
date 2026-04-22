"""Phase 9 — LangGraph node functions with real LLM tool-calling.

Flow: planner (LLM + tools bound) → retriever (execute tool calls) →
      reasoner (check sufficiency) → responder (synthesize answer).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cce.agents.llm import get_llm, get_system_message
from cce.agents.state import AgentState
from cce.config import get_settings
from cce.logging import get_logger

log = get_logger(__name__)


# ── Planner — LLM decides which tools to call ────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """Invoke the LLM (with tools bound). Emits tool_calls or a plain text reply."""
    from cce.agents.tools import ALL_TOOLS  # noqa: PLC0415

    messages = list(state.get("messages", []))
    if not messages:
        messages = [HumanMessage(content=state.get("query", ""))]

    try:
        llm = get_llm().bind_tools(ALL_TOOLS)
        response: AIMessage = llm.invoke([get_system_message()] + messages)
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM call failed (%s) — falling back to direct hybrid search", exc)
        response = _fallback_search(state.get("query", ""))

    return {
        "messages": messages + [response],
        "loop_count": state.get("loop_count", 0) + 1,
    }


def _fallback_search(query: str) -> AIMessage:
    """When no LLM is configured, run hybrid search directly."""
    from cce.retrieval.tools import search_code  # noqa: PLC0415
    try:
        hits = search_code(query, mode="hybrid", k=5)
        lines = [f"- {h.node.qualified_name if h.node else h.path} ({h.path}:{h.line_start})" for h in hits]
        content = "Top results:\n" + "\n".join(lines) if lines else "No results found."
    except Exception:  # noqa: BLE001
        content = "Could not run search. Ensure the index is built with `cce index <path>`."
    return AIMessage(content=content)


# ── Retriever — execute tool calls ───────────────────────────────────────────

def retriever_node(state: AgentState) -> AgentState:
    """Execute all tool_calls from the last AI message; return ToolMessages."""
    from cce.agents.tools import ALL_TOOLS  # noqa: PLC0415

    messages = list(state.get("messages", []))
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {}

    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages: list[ToolMessage] = []
    retrieved: list = list(state.get("retrieved_context", []))

    for call in last.tool_calls:
        tool = tool_map.get(call["name"])
        try:
            result = tool.invoke(call["args"]) if tool else f"Unknown tool: {call['name']}"
            if isinstance(result, list):
                retrieved.extend(result)
        except Exception as exc:  # noqa: BLE001
            result = f"Tool error: {exc}"
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"messages": messages + tool_messages, "retrieved_context": retrieved}


# ── Reasoner — decide if we have enough context ───────────────────────────────

def reasoner_node(state: AgentState) -> AgentState:
    """Append a reasoning step; loop gate handled by should_continue."""
    steps = list(state.get("reasoning_steps", []))
    steps.append(f"loop {state.get('loop_count', 0)}: {len(state.get('retrieved_context', []))} items retrieved")
    return {"reasoning_steps": steps}


# ── Responder — synthesize final answer ───────────────────────────────────────

_MAX_CONTEXT_WORDS = 15_000   # hard safety ceiling for the full message history


def responder_node(state: AgentState) -> AgentState:
    """Use LLM to produce a final answer with citations from retrieved context.

    Before calling the LLM, trims retrieved_context to max_context_items and
    guards against the message history exceeding _MAX_CONTEXT_WORDS.
    """
    # ── Budget: trim retrieved_context ────────────────────────────────────────
    max_items: int = state.get("max_context_items", 20)
    context = state.get("retrieved_context", [])
    if len(context) > max_items:
        log.debug("Trimming retrieved_context from %d → %d items", len(context), max_items)
        context = context[:max_items]

    # ── Budget: trim message history ───────────────────────────────────────────
    messages = list(state.get("messages", []))
    total_words = sum(len(str(getattr(m, "content", "")).split()) for m in messages)
    if total_words > _MAX_CONTEXT_WORDS:
        # Keep system + first human + last N messages that fit
        kept: list = []
        budget = _MAX_CONTEXT_WORDS
        for msg in reversed(messages):
            w = len(str(getattr(msg, "content", "")).split())
            if budget - w < 0:
                break
            kept.insert(0, msg)
            budget -= w
        messages = kept
        log.debug("Trimmed message history to %d messages (%d words budget remaining)",
                  len(messages), budget)

    try:
        llm = get_llm()
        final: AIMessage = llm.invoke([get_system_message()] + messages)
        answer = final.content
    except Exception:  # noqa: BLE001
        answer = messages[-1].content if messages else "No answer."

    # Build citations from retrieved context
    citations = []
    for item in context[:10]:
        if hasattr(item, "node") and item.node:
            citations.append({"symbol": item.node.qualified_name,
                               "file": item.node.file_path,
                               "line": item.node.line_start})
        elif isinstance(item, dict) and "qualified_name" in item:
            citations.append({"symbol": item["qualified_name"], "file": item.get("file_path", "")})

    return {"answer": answer, "citations": citations, "messages": messages,
            "retrieved_context": context}


# ── Edge predicate ────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Return 'retrieve' if LLM wants tools, 'respond' when done."""
    settings = get_settings()
    if state.get("loop_count", 0) >= settings.agent.max_retrieval_loops:
        return "respond"
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "retrieve"
    return "respond"
