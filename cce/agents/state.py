"""LangGraph shared state for the multi-agent runtime.

Budget guards prevent context-window explosion on long retrieval loops:
  - loop_count is hard-capped at max_retrieval_loops (default 3 from Settings)
  - retrieved_context is trimmed to max_context_items (default 20) before the
    responder builds its LLM prompt, preventing unbounded growth
  - context_word_count tracks approximate token usage (trimmed at ~15k words)
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from cce.retrieval.tools import Hit


class RetrievalPlan(TypedDict, total=False):
    """Planner output: which tool calls to issue."""

    tool: str
    args: dict


class AgentState(TypedDict, total=False):
    """Shared state flowing between nodes in the LangGraph StateGraph."""

    # ── Conversation (LangGraph-managed reducer) ──────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── User inputs ───────────────────────────────────────────────────────────
    query: str

    # ── Planner → Retriever ───────────────────────────────────────────────────
    plan: list[RetrievalPlan]

    # ── Retriever output (trimmed before LLM call) ────────────────────────────
    retrieved_context: list[Hit]

    # ── Reasoner bookkeeping ──────────────────────────────────────────────────
    reasoning_steps: list[str]
    loop_count: int
    tool_errors: int
    # F15: coverage axes from the last reasoner turn
    coverage_axes: dict  # {has_subject_symbol, has_symbol_body, has_callers}

    # ── Budget controls ───────────────────────────────────────────────────────
    # Populated from Settings.agent at graph-creation time; both have defaults.
    max_retrieval_loops: int       # hard planner-loop cap (default: 3)
    max_context_items: int         # max items kept in retrieved_context (default: 20)
    context_word_count: int        # running word-count; responder trims if > 15k

    # ── Final answer ──────────────────────────────────────────────────────────
    answer: str
    citations: list[dict]
