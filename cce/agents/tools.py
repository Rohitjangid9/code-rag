"""LangChain Tool wrappers over the typed retrieval surface.

These are the tools the LLM-powered agent nodes can invoke. Each @tool call
delegates to the stable functions defined in `cce.retrieval.tools`. When those
functions are implemented in later phases, the agent gains real capability
without any change to the LangGraph wiring.
"""

from __future__ import annotations

from langchain_core.tools import tool

from cce.retrieval import tools as rt


@tool
def search_code(query: str, mode: str = "hybrid", k: int = 10) -> list[dict]:
    """Search the codebase. mode = lexical | semantic | hybrid | auto."""
    hits = rt.search_code(query=query, mode=mode, k=k)  # type: ignore[arg-type]
    return [h.model_dump() for h in hits]


@tool
def get_symbol(qualified_name: str) -> dict:
    """Return the full record for a symbol given its qualified name."""
    return rt.get_symbol(qualified_name).model_dump()


@tool
def get_file_outline(path: str) -> list[dict]:
    """List all symbols defined in a given file path (repo-relative)."""
    return [n.model_dump() for n in rt.get_file_outline(path)]


@tool
def find_callers(qualified_name: str) -> list[dict]:
    """Return all symbols that call the given symbol."""
    return [n.model_dump() for n in rt.find_callers(qualified_name)]


@tool
def find_references(qualified_name: str) -> list[dict]:
    """Return all source locations that reference the given symbol."""
    return [loc.model_dump() for loc in rt.find_references(qualified_name)]


@tool
def get_neighborhood(qualified_name: str, depth: int = 2) -> dict:
    """Return an N-hop subgraph around the given symbol."""
    return rt.get_neighborhood(qualified_name, depth=depth).model_dump()


@tool
def get_route(pattern_or_path: str) -> dict:
    """Resolve an HTTP URL pattern to its handler + models (Django/FastAPI)."""
    return rt.get_route(pattern_or_path).model_dump()


@tool
def get_component_tree(component_name: str) -> dict:
    """Return the render tree, hooks, and props of a React component."""
    return rt.get_component_tree(component_name).model_dump()


@tool
def get_api_flow(route_or_component: str) -> dict:
    """Return the UI → API → handler → model flow for a given anchor."""
    return rt.get_api_flow(route_or_component).model_dump()


# ── P0-2 / P0-3: deterministic enumeration + grep ─────────────────────────────

@tool
def list_symbols(
    file_path: str | None = None,
    kind: str | None = None,
    name_prefix: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Enumerate symbols deterministically (no ranking).

    Use this — not search_code — when the question is "list every X in file Y"
    or "show me all classes / routes / functions". Filters: file_path (exact or
    suffix), kind (Function | Method | Class | Route | URLPattern | Component |
    PydanticModel | …), name_prefix. Returns up to *limit* rows ordered by
    file_path, line_start.
    """
    return [n.model_dump() for n in rt.list_symbols(
        file_path=file_path, kind=kind, name_prefix=name_prefix, limit=limit,
    )]


@tool
def list_routes(framework: str | None = None) -> list[dict]:
    """List every HTTP route/URL pattern discovered by the framework extractors.

    Optional framework filter: "fastapi" | "django" | "drf". Prefer this over
    search_code when asked "what endpoints does X expose" — it returns every
    route, not just the top-ranked few.
    """
    return [r.model_dump() for r in rt.list_routes(framework=framework)]


@tool
def list_files(glob: str | None = None, limit: int = 2000) -> list[str]:
    """List every indexed file path. Optional shell-style *glob* filter."""
    return rt.list_files(glob=glob, limit=limit)


@tool
def list_cli_commands() -> list[dict]:
    """List Typer/Click CLI commands (functions defined in cli.py / __main__.py)."""
    return [n.model_dump() for n in rt.list_cli_commands()]


@tool
def grep_code(
    pattern: str,
    path_glob: str | None = None,
    limit: int = 50,
    case_sensitive: bool = True,
) -> list[dict]:
    """Regex line-grep over every indexed file's content.

    Use this as a fallback when find_callers returns nothing (the reference may
    be a callback / decorator / first-class identifier the resolver missed).
    Returns (path, line, text) tuples — line numbers are real and verifiable.
    """
    return [h.model_dump() for h in rt.grep_code(
        pattern=pattern, path_glob=path_glob, limit=limit, case_sensitive=case_sensitive,
    )]


ALL_TOOLS = [
    search_code,
    get_symbol,
    get_file_outline,
    find_callers,
    find_references,
    get_neighborhood,
    get_route,
    get_component_tree,
    get_api_flow,
    list_symbols,
    list_routes,
    list_files,
    list_cli_commands,
    grep_code,
]
