"""Phase 9 — LLM factory: OpenAI, Anthropic, Ollama backends.

Selected via CCE_AGENT__LLM_PROVIDER (openai | anthropic | ollama).
Falls back gracefully if the provider's package is not installed.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from cce.config import get_settings
from cce.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are Code Context Engine — an expert code analysis assistant.
You have access to tools that index and query a codebase.

{repo_context}
Tool-selection playbook (follow this before ranked search):
- "list / enumerate every X in file Y" → call `list_symbols(file_path=Y, kind=X?)`
  Example: "list all agent nodes" → `list_symbols(file_path="agents/nodes.py", kind="Function")`
- "what HTTP endpoints / routes does Z expose" → call `list_routes(framework?)`
- "what happens when I POST/GET /X" → call `find_route_handler(method="POST", path="/X")`
- "what CLI commands does this project have" → call `list_cli_commands`
- When the user says "all / every / every single", pass `limit=500` to `list_*` tools.
- "what files are in this repo / matching Y" → call `list_files(glob=Y)`
- "who calls X / where is X used" → call `find_callers(X)` FIRST;
   if it returns nothing, fall back to `grep_code(pattern="\\\\bX\\\\b")`.
- "show me function X at line N" / "definition of X" → `get_symbol(qname)`
  or `get_file_outline(path)` for all symbols in a file.
  If the answer needs the function body, call `get_file_slice(path, start, end)` next.
- Only use `search_code` for fuzzy / conceptual questions ("how does auth
  work", "where is retrieval ranked"). It is ranked top-k and will MISS
  entries — never rely on it for exhaustive lists.

When answering:
1. Use the tool that matches the question shape (see playbook).
2. Always cite the qualified symbol name and file:line from tool results —
   do NOT invent line numbers.
3. If you need more context, call tools again (max {max_loops} loops).
4. Be concise. Prefer code references over explanations.
5. If a symbol is not found, say so clearly.
"""

RESPONDER_SYSTEM_PROMPT = """\
You are Code Context Engine — an expert code analysis assistant.
Using ONLY the retrieved context and CITATION TABLE below, answer the user.

Rules:
1. Every `file:line` or symbol citation in your answer MUST appear verbatim
   in the CITATION TABLE. Do not invent file paths, line numbers, or symbol
   names — if the table doesn't contain it, don't cite it.
2. Use the format `qualified.name (file:line_start-line_end)` when citing a
   symbol, exactly as written in the table.
3. If the retrieved context does not answer the question, say so plainly.
4. Be concise. Prefer code references over prose.
5. Do not call tools at this stage; retrieval is complete.
"""


def _ensure_api_key_loaded() -> None:
    """Load .env if OPENAI_API_KEY is missing from the real OS environment.

    pydantic-settings skips variables without the CCE_ prefix, so the key
    may exist in .env but not be visible to os.getenv().
    """
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
            load_dotenv(".env")
        except Exception:
            pass


def _build_llm(model: str) -> BaseChatModel:
    """Instantiate a chat model for the configured provider and *model* name.

    F37: raises ``RuntimeError`` when the engine is in offline mode.
    """
    settings = get_settings()
    if settings.offline:
        raise RuntimeError(
            "CCE is running in offline mode (CCE_OFFLINE=true). "
            "LLM calls are disabled. Use `cce answer` with a local Ollama model "
            "or disable offline mode."
        )
    cfg = settings.agent
    provider = cfg.llm_provider
    temp = cfg.llm_temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        _ensure_api_key_loaded()
        log.info("Using OpenAI LLM: %s", model)
        return ChatOpenAI(model=model, temperature=temp, streaming=True)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        log.info("Using Anthropic LLM: %s", model)
        return ChatAnthropic(model=model, temperature=temp, streaming=True)  # type: ignore[call-arg]

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama  # noqa: PLC0415
        log.info("Using Ollama LLM: %s", model)
        return ChatOllama(model=model, temperature=temp)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Set CCE_AGENT__LLM_PROVIDER to openai | anthropic | ollama."
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the base chat model (falls back for callers that haven't migrated).

    Prefer :func:`get_planner_llm` / :func:`get_responder_llm` for new code.
    """
    cfg = get_settings().agent
    return _build_llm(cfg.llm_model)


@lru_cache(maxsize=1)
def get_planner_llm() -> BaseChatModel:
    """Return the LLM used by the planner node (tool-selection loop).

    Resolved from ``AgentSettings.planner_model``; falls back to ``llm_model``.
    """
    cfg = get_settings().agent
    model = cfg.planner_model or cfg.llm_model
    return _build_llm(model)


@lru_cache(maxsize=1)
def get_responder_llm() -> BaseChatModel:
    """Return the LLM used by the responder node (final answer synthesis).

    Resolved from ``AgentSettings.responder_model``; falls back to ``llm_model``.
    """
    cfg = get_settings().agent
    model = cfg.responder_model or cfg.llm_model
    return _build_llm(model)


def _build_repo_context() -> str:
    """F-M8: read .cce/index.json and render a one-paragraph repo summary.

    Returns an empty string when the manifest is missing or unreadable so
    the prompt still formats cleanly on un-indexed repos.
    """
    import json as _json  # noqa: PLC0415
    settings = get_settings()
    manifest_path = settings.paths.data_dir / "index.json"
    if not manifest_path.exists():
        return ""
    try:
        data = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""

    root = data.get("root") or str(settings.repo_root or "")
    frameworks = data.get("frameworks") or []
    languages = data.get("languages") or []
    file_count = data.get("file_count", 0)
    symbol_count = data.get("symbol_count", 0)

    lines = ["Repo context (from index manifest):"]
    if root:
        lines.append(f"- root: {root}")
    if languages:
        lines.append(f"- languages: {', '.join(languages)}")
    if frameworks:
        lines.append(f"- frameworks: {', '.join(frameworks)}")
    else:
        lines.append("- frameworks: (none detected)")
    if file_count:
        lines.append(f"- files: {file_count}, symbols: {symbol_count}")
    return "\n".join(lines) + "\n"


def get_system_message():
    from langchain_core.messages import SystemMessage  # noqa: PLC0415
    cfg = get_settings().agent
    return SystemMessage(
        content=SYSTEM_PROMPT.format(
            max_loops=cfg.max_retrieval_loops,
            repo_context=_build_repo_context(),
        )
    )


def get_responder_system_message():
    from langchain_core.messages import SystemMessage  # noqa: PLC0415
    return SystemMessage(content=RESPONDER_SYSTEM_PROMPT)
