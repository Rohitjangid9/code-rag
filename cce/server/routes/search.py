"""Search endpoints — lexical (Phase 2) and symbol (Phase 3) implemented."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from cce.retrieval.tools import Hit, search_code

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    mode: Literal["auto", "lexical", "semantic", "hybrid"] = "auto"
    k: int = 10
    filters: dict | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[Hit]


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """BM25 lexical + symbol FTS search. Semantic/hybrid deferred to Phase 8."""
    hits = search_code(query=req.query, mode=req.mode, k=req.k, filters=req.filters)
    return SearchResponse(query=req.query, mode=req.mode, hits=hits)
