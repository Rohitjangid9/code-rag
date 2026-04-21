"""Phase 9 — LLM factory: OpenAI, Anthropic, Ollama backends.

Selected via CCE_AGENT__LLM_PROVIDER (openai | anthropic | ollama).
Falls back gracefully if the provider's package is not installed.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from cce.config import get_settings
from cce.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are Code Context Engine — an expert code analysis assistant.
You have access to tools that index and query a codebase.

When answering:
1. Use tools to find relevant code before answering.
2. Always cite the qualified symbol name and file path.
3. If you need more context, call tools again (max {max_loops} loops).
4. Be concise. Prefer code references over explanations.
5. If a symbol is not found, say so clearly.
"""


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the configured chat model, cached as a singleton."""
    cfg = get_settings().agent
    provider = cfg.llm_provider
    model = cfg.llm_model
    temp = cfg.llm_temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
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


def get_system_message():
    from langchain_core.messages import SystemMessage  # noqa: PLC0415
    cfg = get_settings().agent
    return SystemMessage(content=SYSTEM_PROMPT.format(max_loops=cfg.max_retrieval_loops))
