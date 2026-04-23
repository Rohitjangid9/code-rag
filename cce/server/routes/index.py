"""Indexing endpoints (Phase 1+)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class IndexRequest(BaseModel):
    # F-M7: ``path`` is the repo root; ``.cce/`` is created inside it.
    path: str
    layers: list[str] = ["lexical", "symbols", "graph", "framework"]
    include_dirs: list[str] = []
    skip_dirs: list[str] = []


class IndexStatus(BaseModel):
    path: str
    files_total: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    symbols_indexed: int = 0
    edges_indexed: int = 0
    elapsed_s: float = 0.0
    phase: str = "done"


@router.post("", response_model=IndexStatus)
def start_index(req: IndexRequest) -> IndexStatus:
    """Run the indexing pipeline synchronously for the given repo root."""
    from cce.config import get_settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    p = Path(req.path).resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail=f"path not found: {req.path}")

    settings = get_settings(repo_root=p)
    stats = IndexPipeline(settings=settings).run(
        p,
        layers=req.layers,
        include_dirs=set(req.include_dirs) or None,
        skip_dirs=set(req.skip_dirs) or None,
    )
    return IndexStatus(
        path=str(p),
        files_total=stats.files_total,
        files_new=stats.files_new,
        files_changed=stats.files_changed,
        files_deleted=stats.files_deleted,
        symbols_indexed=stats.symbols_indexed,
        edges_indexed=stats.edges_indexed,
        elapsed_s=stats.elapsed_s,
    )


@router.get("/status", response_model=IndexStatus)
def get_status() -> IndexStatus:
    """Return the current indexer state."""
    raise HTTPException(status_code=501, detail="indexing pipeline lands in Phases 1-7")
