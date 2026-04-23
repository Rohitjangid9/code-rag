"""FastAPI app factory with lifespan and router mounts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cce.config import get_settings
from cce.logging import get_logger
from cce.server import mcp as mcp_module
from cce.server.auth import APIKeyMiddleware, RateLimitMiddleware
from cce.server.routes import agent, graph, health, index, search, symbols

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources (embedder, DB connections) on startup."""
    import json as _json  # noqa: PLC0415
    settings = get_settings()
    log.info("cce server starting (embedder=%s)", settings.embedder.backend)

    # F12: read index manifest and warn if missing or schema_version mismatches.
    manifest_path = settings.paths.data_dir / "index.json"
    if not manifest_path.exists():
        log.warning(
            "Index manifest not found at %s — run `cce index <path>` first",
            manifest_path,
        )
    else:
        try:
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
            mver = manifest.get("schema_version")
            if mver != settings.schema_version:
                log.warning(
                    "Index manifest schema_version mismatch (manifest=%s, expected=%s) "
                    "— consider re-running `cce index <path>`",
                    mver, settings.schema_version,
                )
            else:
                log.info(
                    "Index manifest OK: root=%s indexed_at=%s",
                    manifest.get("root"), manifest.get("indexed_at"),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not parse index manifest: %s", exc)

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

    # F30: rate-limit first (cheapest check), then auth
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyMiddleware)

    # Tighten CORS — only explicit origins when api_keys are configured (F30)
    cors_origins = settings.server.cors_origins
    if settings.server.api_keys and cors_origins == ["*"]:
        cors_origins = []  # block wildcard CORS when auth is enabled
        log.warning("CORS wildcard disabled because API keys are configured; "
                    "set CCE_SERVER__CORS_ORIGINS explicitly.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    app.include_router(health.router)
    app.include_router(index.router, prefix="/index", tags=["index"])
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(symbols.router, prefix="/symbols", tags=["symbols"])
    app.include_router(graph.router, prefix="/graph", tags=["graph"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])
    app.include_router(mcp_module.router)   # POST /mcp  +  GET /mcp/tools

    return app
