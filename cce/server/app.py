"""FastAPI app factory with lifespan and router mounts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cce.config import get_settings
from cce.logging import get_logger
from cce.server import mcp as mcp_module
from cce.server.routes import agent, graph, health, index, search, symbols

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources (embedder, DB connections) on startup."""
    settings = get_settings()
    log.info("cce server starting (embedder=%s)", settings.embedder.backend)
    # Phase 7+: load embedder model here.
    # Phase 3+: open SQLite / Kùzu / Qdrant connections here.
    yield
    log.info("cce server shutting down")


def create_app() -> FastAPI:
    """Application factory (referenced by uvicorn --factory)."""
    settings = get_settings()
    app = FastAPI(
        title="Code Context Engine",
        version="0.1.0",
        description="Hybrid code indexer + LangGraph agent runtime.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(index.router, prefix="/index", tags=["index"])
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(symbols.router, prefix="/symbols", tags=["symbols"])
    app.include_router(graph.router, prefix="/graph", tags=["graph"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])
    app.include_router(mcp_module.router)   # POST /mcp  +  GET /mcp/tools

    return app
