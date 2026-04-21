"""Symbol endpoints — Phases 3 and 4 implemented."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cce.graph.schema import Location, Node
from cce.retrieval.tools import (
    find_callers,
    find_implementations,
    find_references,
    get_file_outline,
    get_symbol,
)

router = APIRouter()


@router.get("/outline", response_model=list[Node])
def outline(path: str) -> list[Node]:
    """List all symbols defined in a file (repo-relative path)."""
    return get_file_outline(path)


@router.get("/callers", response_model=list[Node])
def callers(qname: str) -> list[Node]:
    return find_callers(qname)


@router.get("/refs", response_model=list[Location])
def refs(qname: str) -> list[Location]:
    return find_references(qname)


@router.get("/implementations", response_model=list[Node])
def implementations(qname: str) -> list[Node]:
    return find_implementations(qname)


@router.get("/{qualified_name:path}", response_model=Node)
def symbol(qualified_name: str) -> Node:
    try:
        return get_symbol(qualified_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {qualified_name}")
