"""Liveness / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from cce import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/ready")
def ready() -> dict:
    return {"ready": True}
