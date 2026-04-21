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
]
