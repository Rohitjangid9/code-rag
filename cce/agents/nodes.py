"""Phase 9 — LangGraph node functions with real LLM tool-calling.

Flow:
    START → planner
        ├─ has_tool_calls? yes → retriever → reasoner
        │                                      ├─ continue → planner (loop)
        │                                      └─ stop     → responder → END
        └─ has_tool_calls? no  → responder → END
"""

from __future__ import annotations

import json as _json
import re as _re
import uuid as _uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cce.agents.llm import get_llm, get_planner_llm, get_responder_llm, get_responder_system_message, get_system_message
from cce.agents.state import AgentState
from cce.agents.trace import emit as _trace_emit, start_timer as _trace_timer
from cce.config import get_settings
from cce.logging import get_logger
from cce.telemetry import get_tracer

log = get_logger(__name__)

_MAX_CONTEXT_WORDS = 15_000   # hard safety ceiling for the full message history

# Signatures that indicate the LLM emitted a tool-call preamble as plain text
# instead of structured tool_calls (common with Ollama / open-weights models
# whose tool-calling schema doesn't match langchain's binding layer).
_TOOL_LEAK_PATTERNS = (
    "to=functions.",
    "to=multi_tool_use.",
    "<|tool_calls|>",
    "<|tool_call|>",
    '"tool_uses"',
    '"recipient_name"',
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trim_messages(messages: list, budget: int = _MAX_CONTEXT_WORDS) -> list:
    """Trim the message history from the oldest end until it fits *budget* words."""
    total = sum(len(str(getattr(m, "content", "")).split()) for m in messages)
    if total <= budget:
        return messages
    kept: list = []
    remaining = budget
    for msg in reversed(messages):
        w = len(str(getattr(msg, "content", "")).split())
        if remaining - w < 0:
            break
        kept.insert(0, msg)
        remaining -= w
    return kept


def _hit_key(item) -> tuple:
    """Stable dedup key for a retrieved-context entry (dict or Hit)."""
    if isinstance(item, dict):
        node = item.get("node") or {}
        if isinstance(node, dict) and node.get("id"):
            return ("id", node["id"])
        return ("loc", item.get("path", ""), item.get("line_start", 0))
    node = getattr(item, "node", None)
    if node is not None and getattr(node, "id", None):
        return ("id", node.id)
    return ("loc", getattr(item, "path", ""), getattr(item, "line_start", 0))


def _dedup_context(items: list) -> list:
    seen: set = set()
    out: list = []
    for it in items:
        k = _hit_key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _format_context_block(items: list, max_items: int = 20) -> str:
    """Render retrieved_context into a compact text block for the responder prompt."""
    lines: list[str] = []
    for i, item in enumerate(items[:max_items], 1):
        if isinstance(item, dict):
            node = item.get("node") or {}
            sym = (node.get("qualified_name") if isinstance(node, dict) else None) or item.get("path", "")
            file = (node.get("file_path") if isinstance(node, dict) else None) or item.get("path", "")
            line = (node.get("line_start") if isinstance(node, dict) else None) or item.get("line_start", 0)
            snip = item.get("snippet") or (node.get("signature", "") if isinstance(node, dict) else "") or ""
        else:
            n = getattr(item, "node", None)
            sym = getattr(n, "qualified_name", None) if n else getattr(item, "path", "")
            file = getattr(n, "file_path", None) if n else getattr(item, "path", "")
            line = getattr(n, "line_start", None) if n else getattr(item, "line_start", 0)
            snip = getattr(item, "snippet", "") or ""
        lines.append(f"[{i}] {sym} ({file}:{line}) — {str(snip)[:160]}")
    return "\n".join(lines)


# ── P0-1: tool-call leak parsing ──────────────────────────────────────────────

def _content_looks_like_tool_leak(content: str) -> bool:
    if not content:
        return False
    head = content[:400]
    return any(sig in head for sig in _TOOL_LEAK_PATTERNS)


def _parse_leaked_tool_calls(content: str, known_tool_names: set[str]) -> list[dict]:
    """Best-effort extractor for tool-call preambles written as plain text.

    Handles three common shapes seen from non-OpenAI models:
        1. `to=functions.NAME {"k":"v"}`
        2. `{"tool_uses":[{"recipient_name":"functions.NAME","parameters":{...}}]}`
        3. `<|tool_call|>{"name":"NAME","arguments":{...}}`

    Returns a list of {name, args, id} dicts suitable for AIMessage.tool_calls.
    Garbled / non-ASCII sentinel tokens between markers are tolerated.
    """
    calls: list[dict] = []

    # Shape 1: to=functions.NAME  ...  {json}
    for m in _re.finditer(r"to=(?:functions|multi_tool_use)\.([A-Za-z_][A-Za-z0-9_]*)", content):
        name = m.group(1)
        if name == "parallel":
            continue  # handled by shape 2 below
        args = _extract_first_json_object(content, start=m.end())
        if name in known_tool_names and isinstance(args, dict):
            calls.append({"name": name, "args": args, "id": f"leak-{_uuid.uuid4().hex[:8]}"})

    # Shape 2: multi_tool_use.parallel with tool_uses array. Walk every
    # balanced `{...}` until one parses AND contains a top-level "tool_uses".
    idx = 0
    while '"tool_uses"' in content[idx:]:
        obj = _extract_first_json_object(content, start=idx)
        if obj is None:
            break
        if isinstance(obj, dict) and "tool_uses" in obj:
            for use in obj.get("tool_uses", []) or []:
                rn = str(use.get("recipient_name", ""))
                name = rn.rsplit(".", 1)[-1] if "." in rn else rn
                params = use.get("parameters") or use.get("args") or {}
                if name in known_tool_names and isinstance(params, dict):
                    calls.append({"name": name, "args": params,
                                  "id": f"leak-{_uuid.uuid4().hex[:8]}"})
            break
        next_brace = content.find("{", idx + 1)
        if next_brace == -1:
            break
        idx = next_brace

    # Shape 3: explicit {"name": ..., "arguments": ...}
    for m in _re.finditer(r'\{"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"', content):
        name = m.group(1)
        obj = _extract_first_json_object(content, start=m.start())
        if name in known_tool_names and isinstance(obj, dict):
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(args, dict):
                calls.append({"name": name, "args": args,
                              "id": f"leak-{_uuid.uuid4().hex[:8]}"})

    # Dedup on (name, sorted_args_json)
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for c in calls:
        key = (c["name"], _json.dumps(c["args"], sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _extract_first_json_object(text: str, start: int = 0) -> dict | None:
    """Return the first balanced `{...}` JSON object at/after *start*, else None."""
    i = text.find("{", start)
    while i != -1 and i < len(text):
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i: j + 1]
                    try:
                        return _json.loads(candidate)
                    except Exception:  # noqa: BLE001
                        break
        i = text.find("{", i + 1)
    return None


def _debug_log_ai_message(response: AIMessage) -> None:
    """Log raw AIMessage fields when CCE_AGENT__DEBUG=true."""
    if not get_settings().agent.debug:
        return
    content_head = (getattr(response, "content", "") or "")[:300]
    log.info(
        "planner AIMessage: tool_calls=%d content_head=%r",
        len(getattr(response, "tool_calls", []) or []),
        content_head,
    )


# ── F16: Tool-result summarization ───────────────────────────────────────────

def _summarize_old_tool_messages(messages: list) -> list:
    """Replace all but the last ToolMessage with a compact one-line summary.

    Keeps the most-recent ToolMessage verbatim (the planner needs fresh detail)
    and compresses older ones to ``"<tool>: N hits, top=[…]"`` so the context
    window stops growing unboundedly across loops.
    """
    # Identify indices of ToolMessage objects
    tool_idxs = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_idxs) <= 1:
        return messages  # nothing to compress

    to_compress = set(tool_idxs[:-1])  # keep last one intact
    out: list = []
    for i, msg in enumerate(messages):
        if i not in to_compress:
            out.append(msg)
            continue
        # Build a one-line summary from the ToolMessage content
        raw = getattr(msg, "content", "") or ""
        try:
            data = _json.loads(raw)
            if isinstance(data, list):
                summary = f"{len(data)} hits, top={[str(d)[:60] for d in data[:3]]}"
            elif isinstance(data, dict):
                status = data.get("status", "ok")
                summary = f"status={status} keys={list(data.keys())[:5]}"
            else:
                summary = str(raw)[:120]
        except Exception:  # noqa: BLE001
            summary = str(raw)[:120]
        out.append(ToolMessage(
            content=f"(summarised) {summary}",
            tool_call_id=getattr(msg, "tool_call_id", ""),
        ))
    return out


# ── Planner — LLM decides which tools to call ────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """Invoke the LLM (with tools bound). Emits tool_calls or a plain text reply."""
    from cce.agents.tools import ALL_TOOLS  # noqa: PLC0415

    _t = _trace_timer()
    thread_id = state.get("query", "")[:32]
    turn = state.get("loop_count", 0)

    messages = list(state.get("messages", []))
    if not messages:
        messages = [HumanMessage(content=state.get("query", ""))]
    messages = _trim_messages(messages)

    # F16: summarize older ToolMessages to keep context window small
    messages = _summarize_old_tool_messages(messages)

    fallback_hits: list = []
    error_detail: str | None = None
    with get_tracer().start_as_current_span("cce.planner") as span:  # F33
        span.set_attribute("thread_id", thread_id)
        span.set_attribute("turn", turn)
        try:
            llm = get_planner_llm().bind_tools(ALL_TOOLS)
            response: AIMessage = llm.invoke([get_system_message()] + messages)
        except Exception as exc:  # noqa: BLE001
            error_detail = str(exc)
            span.record_exception(exc)
            log.warning("LLM call failed (%s) — falling back to direct hybrid search", exc)
            response, fallback_hits = _fallback_search(state.get("query", ""))

    # P0-1: tool-call sanity — recover tool calls leaked into content.
    _debug_log_ai_message(response)
    if (not getattr(response, "tool_calls", None)
            and _content_looks_like_tool_leak(getattr(response, "content", "") or "")):
        known = {t.name for t in ALL_TOOLS}
        recovered = _parse_leaked_tool_calls(response.content, known)
        if recovered:
            log.warning("Recovered %d leaked tool_calls from content preamble", len(recovered))
            response = AIMessage(content="", tool_calls=recovered)
        else:
            log.warning("Tool-call leak detected but no recoverable calls — stripping preamble")
            response = AIMessage(
                content="(planner emitted malformed tool-call text; no retrieval this turn)"
            )

    _trace_emit({
        "node": "planner", "thread_id": thread_id, "turn": turn,
        "elapsed_ms": _t(), **({"error": error_detail} if error_detail else {}),
    })

    update: dict = {
        "messages": [response],
        "loop_count": state.get("loop_count", 0) + 1,
    }
    if fallback_hits:
        merged = list(state.get("retrieved_context", [])) + fallback_hits
        update["retrieved_context"] = _dedup_context(merged)
    return update


def _fallback_search(query: str) -> tuple[AIMessage, list]:
    """When no LLM is configured, run hybrid search directly and surface hits."""
    from cce.retrieval.tools import search_code  # noqa: PLC0415
    try:
        hits = search_code(query, mode="hybrid", k=5)
        lines = [
            f"- {h.node.qualified_name if h.node else h.path} ({h.path}:{h.line_start})"
            for h in hits
        ]
        content = "Top results:\n" + "\n".join(lines) if lines else "No results found."
        return AIMessage(content=content), [h.model_dump() for h in hits]
    except Exception:  # noqa: BLE001
        return (
            AIMessage(content="Could not run search. Ensure the index is built with `cce index <path>`."),
            [],
        )


# ── Retriever — execute tool calls ───────────────────────────────────────────

def retriever_node(state: AgentState) -> AgentState:
    """Execute all tool_calls from the last AI message; return ToolMessages + hits."""
    from cce.agents.tools import ALL_TOOLS  # noqa: PLC0415

    messages = list(state.get("messages", []))
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {}

    thread_id = state.get("query", "")[:32]
    turn = state.get("loop_count", 0)
    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages: list[ToolMessage] = []
    retrieved: list = list(state.get("retrieved_context", []))

    for call in last.tool_calls:
        _t = _trace_timer()
        tool = tool_map.get(call["name"])
        error_detail: str | None = None
        hits_count = 0
        try:
            result = tool.invoke(call["args"]) if tool else f"Unknown tool: {call['name']}"
            if isinstance(result, list):
                hits_count = len(result)
                retrieved.extend(result)
        except Exception as exc:  # noqa: BLE001
            error_detail = str(exc)
            result = {"status": "error", "error_class": type(exc).__name__, "message": str(exc)}
        content = _json.dumps(result, default=str) if isinstance(result, dict) else str(result)
        tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        _trace_emit({
            "node": "retriever", "thread_id": thread_id, "turn": turn,
            "tool": call["name"], "args": call.get("args", {}),
            "hits": hits_count, "elapsed_ms": _t(),
            **({"error": error_detail} if error_detail else {}),
        })

    return {"messages": tool_messages, "retrieved_context": _dedup_context(retrieved)}


# ── Reasoner — decide if we have enough context ───────────────────────────────

def reasoner_node(state: AgentState) -> AgentState:
    """Append a reasoning step; loop gate handled by should_continue.

    F15: scores the retrieved context on three coverage axes and forces another
    planning loop when any required axis is missing and retries remain.
    """
    _t = _trace_timer()
    thread_id = state.get("query", "")[:32]
    turn = state.get("loop_count", 0)

    steps = list(state.get("reasoning_steps", []))
    context = state.get("retrieved_context", [])
    steps.append(f"loop {turn}: {len(context)} items retrieved")

    # Count structured tool errors from the last retriever turn
    tool_errors = 0
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage):
            content = getattr(msg, "content", "") or ""
            if content.startswith('{"status": "error"'):
                tool_errors += 1

    if tool_errors:
        steps.append(f"tools returned {tool_errors} error(s) — retrying")

    # F15: coverage-axis scoring
    has_subject_symbol = any(
        (isinstance(it, dict) and (it.get("node") or {}).get("qualified_name"))
        or (not isinstance(it, dict) and getattr(getattr(it, "node", None), "qualified_name", None))
        for it in context
    )
    has_symbol_body = any(
        (isinstance(it, dict) and it.get("snippet") and len(str(it.get("snippet", ""))) > 50)
        or (not isinstance(it, dict) and len(getattr(it, "snippet", "") or "") > 50)
        for it in context
    )
    has_callers = len(context) >= 2  # rough proxy: ≥2 hits suggests caller context

    coverage = {
        "has_subject_symbol": has_subject_symbol,
        "has_symbol_body": has_symbol_body,
        "has_callers": has_callers,
    }
    missing_axes = [k for k, v in coverage.items() if not v]
    if missing_axes:
        steps.append(f"coverage gaps: {', '.join(missing_axes)}")

    _trace_emit({
        "node": "reasoner", "thread_id": thread_id, "turn": turn,
        "tool_errors": tool_errors, "coverage": coverage,
        "elapsed_ms": _t(),
    })

    return {"reasoning_steps": steps, "tool_errors": tool_errors, "coverage_axes": coverage}


# ── Responder — synthesize final answer ───────────────────────────────────────

def _extract_node_fields(item) -> tuple[str, str, int, int]:
    """Return (qname, file, line_start, line_end) from a Hit dict or object."""
    if isinstance(item, dict):
        node = item.get("node") or {}
        if isinstance(node, dict) and node.get("qualified_name"):
            return (
                str(node.get("qualified_name", "")),
                str(node.get("file_path", "")),
                int(node.get("line_start", 0) or 0),
                int(node.get("line_end", 0) or 0),
            )
        return (
            str(item.get("qualified_name") or item.get("path", "")),
            str(item.get("file_path") or item.get("path", "")),
            int(item.get("line_start", 0) or 0),
            int(item.get("line_end", 0) or 0),
        )
    n = getattr(item, "node", None)
    if n is not None:
        return (
            getattr(n, "qualified_name", "") or "",
            getattr(n, "file_path", "") or "",
            int(getattr(n, "line_start", 0) or 0),
            int(getattr(n, "line_end", 0) or 0),
        )
    return (
        getattr(item, "path", "") or "",
        getattr(item, "path", "") or "",
        int(getattr(item, "line_start", 0) or 0),
        int(getattr(item, "line_end", 0) or 0),
    )


def _build_citation_table(context: list, limit: int = 20) -> list[dict]:
    """Stable, de-duplicated citation list derived from retrieved_context."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for item in context[:limit]:
        sym, file, ls, le = _extract_node_fields(item)
        if not file:
            continue
        key = (sym, file, ls, le)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": len(out) + 1, "symbol": sym, "file": file,
                    "line_start": ls, "line_end": le})
    return out


def _format_citation_table(citations: list[dict]) -> str:
    if not citations:
        return "(citation table is empty — no verified references available)"
    lines = []
    for c in citations:
        span = f"{c['line_start']}-{c['line_end']}" if c["line_end"] else str(c["line_start"])
        lines.append(f"[{c['id']}] {c['symbol']} ({c['file']}:{span})")
    return "\n".join(lines)


# Match a plausible file:line citation (path + extension + colon + 1+ digits).
_CITE_RX = _re.compile(
    r"([\w./\\-]+\.(?:py|js|jsx|ts|tsx|mjs|cjs|java|go|rb|rs|php|cs|cpp|c|h|hpp))"
    r":(\d+)(?:-(\d+))?",
)


def _validate_citations(answer: str, citations: list[dict]) -> tuple[str, list[str]]:
    """Strip unverifiable `file:line` citations from *answer*.

    Returns (sanitised_answer, list_of_dropped_citations). A citation is valid
    iff an entry in *citations* has the same file AND a line range containing
    the cited line.
    """
    if not answer:
        return answer, []
    valid_by_file: dict[str, list[tuple[int, int]]] = {}
    for c in citations:
        valid_by_file.setdefault(c["file"], []).append(
            (c["line_start"], c["line_end"] or c["line_start"])
        )

    dropped: list[str] = []

    def _ok(file: str, line: int) -> bool:
        # Accept exact file match or repo-suffix match (handles index-root drift).
        for f, ranges in valid_by_file.items():
            if file == f or f.endswith("/" + file) or file.endswith("/" + f):
                for ls, le in ranges:
                    # Whole-file citation fallback (lexical hits with no line range).
                    if ls == 0 and le == 0:
                        return True
                    if ls <= line <= max(le, ls):
                        return True
        return False

    def _sub(m: _re.Match) -> str:
        file, line_s, _end = m.group(1), m.group(2), m.group(3)
        try:
            line_i = int(line_s)
        except ValueError:
            return m.group(0)
        if _ok(file, line_i):
            return m.group(0)
        dropped.append(f"{file}:{line_s}")
        return f"{file}:?"  # marker so the reader can see a citation was removed

    sanitised = _CITE_RX.sub(_sub, answer)
    return sanitised, dropped


def responder_node(state: AgentState) -> AgentState:
    """Use LLM to produce a final answer grounded in retrieved_context.

    P0-4: Builds a CITATION TABLE of verified (symbol, file, line) triples and
    instructs the LLM to cite only from it. After generation, any ``file:line``
    reference that isn't backed by the table is stripped.
    """
    max_items: int = state.get("max_context_items", 20)
    context = list(state.get("retrieved_context", []))
    if len(context) > max_items:
        log.debug("Trimming retrieved_context from %d → %d items", len(context), max_items)
        context = context[:max_items]

    messages = _trim_messages(list(state.get("messages", [])))

    context_block = _format_context_block(context, max_items=max_items)
    citations = _build_citation_table(context, limit=max_items)
    citation_block = _format_citation_table(citations)

    prompt: list = [get_responder_system_message()]
    if context_block:
        prompt.append(SystemMessage(content=f"Retrieved context:\n{context_block}"))
    else:
        prompt.append(SystemMessage(content="Retrieved context: (none — no tools produced results)"))
    prompt.append(SystemMessage(content=f"CITATION TABLE (the only citations you may use):\n{citation_block}"))
    prompt.extend(messages)

    with get_tracer().start_as_current_span("cce.responder") as span:  # F33
        span.set_attribute("context_items", len(context))
        try:
            llm = get_responder_llm()
            final: AIMessage = llm.invoke(prompt)
            answer = final.content
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            answer = messages[-1].content if messages else "No answer."

    if get_settings().agent.strict_citations and isinstance(answer, str):
        sanitised, dropped = _validate_citations(answer, citations)
        if dropped:
            log.warning("Dropped %d unverifiable citations: %s", len(dropped), dropped[:5])
        answer = sanitised

    return {
        "answer": answer,
        "citations": [{"symbol": c["symbol"], "file": c["file"], "line": c["line_start"]}
                      for c in citations[:10]],
        "retrieved_context": context,
    }


# ── Edge predicates ───────────────────────────────────────────────────────────

def has_tool_calls(state: AgentState) -> str:
    """Route planner output: 'retrieve' if the last AI message emitted tool_calls."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "retrieve"
    return "respond"


def should_continue(state: AgentState) -> str:
    """After retriever+reasoner: loop back to planner unless loop cap is reached.

    F15: forces another loop when coverage axes are incomplete (no subject symbol,
    no body, no callers) and retries remain.  Always continues when there were tool errors so the planner
    can recover.  Exits early once ALL axes are satisfied so we don't burn the
    remaining loop budget on redundant retrieval.
    """
    settings = get_settings()
    loop_count = state.get("loop_count", 0)
    max_loops = settings.agent.max_retrieval_loops
    if loop_count >= max_loops:
        return "respond"
    # Tool errors — must retry (unless we already hit max_loops above)
    if state.get("tool_errors", 0) > 0:
        return "plan"
    # F15: coverage-axis check — loop if any axis is still missing
    axes = state.get("coverage_axes") or {}
    if axes and not all(axes.values()):
        return "plan"
    # All axes satisfied (or no axes tracked yet on first loop) → respond early
    # rather than running empty retrieval rounds.
    return "respond"
