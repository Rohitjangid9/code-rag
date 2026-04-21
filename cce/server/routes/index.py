"""Indexing endpoints (Phase 1+)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class IndexRequest(BaseModel):
    path: str
    layers: list[str] = ["lexical", "symbols", "graph", "semantic"]


class IndexStatus(BaseModel):
    path: str
    files_total: int = 0
    files_indexed: int = 0
    phase: str = "idle"


@router.post("", response_model=IndexStatus)
def start_index(req: IndexRequest) -> IndexStatus:
    """Kick off an indexing job for the given path."""
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {req.path}")
    raise HTTPException(status_code=501, detail="indexing pipeline lands in Phases 1-7")


@router.get("/status", response_model=IndexStatus)
def get_status() -> IndexStatus:
    """Return the current indexer state."""
    raise HTTPException(status_code=501, detail="indexing pipeline lands in Phases 1-7")
