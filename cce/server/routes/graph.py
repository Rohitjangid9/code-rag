"""Graph endpoints — Phase 5 implemented; Phase 6+ stubs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cce.retrieval.tools import (
    ComponentTree,
    CrossStackFlow,
    RouteInfo,
    SubGraph,
    get_neighborhood,
    get_route,
    get_component_tree,
    get_api_flow,
)

router = APIRouter()


@router.get("/neighborhood", response_model=SubGraph)
def neighborhood(qname: str, depth: int = 2) -> SubGraph:
    """Return an N-hop subgraph around a symbol (Phase 5)."""
    try:
        return get_neighborhood(qname, depth=depth)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {qname}")


@router.get("/route", response_model=RouteInfo)
def route(pattern: str) -> RouteInfo:
    """Resolve a URL pattern to its handler (Phase 6)."""
    try:
        return get_route(pattern)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="framework extractors land in Phase 6")


@router.get("/component", response_model=ComponentTree)
def component(name: str) -> ComponentTree:
    """React component tree (Phase 6)."""
    try:
        return get_component_tree(name)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="React extractor lands in Phase 6")


@router.get("/api-flow", response_model=CrossStackFlow)
def api_flow(anchor: str) -> CrossStackFlow:
    """Cross-stack UI→API flow (Phase 6)."""
    try:
        return get_api_flow(anchor)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="cross-stack flow lands in Phase 6")
