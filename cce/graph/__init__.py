"""Layer 3 — code graph store (SQLite default; DuckDB/Postgres via F34 factory)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_graph_store():
    """Return the configured graph store instance (F34).

    Backend is selected via ``CCE_STORE__GRAPH_BACKEND`` (default: sqlite).
    The result is cached as a singleton for the process lifetime.
    """
    from cce.config import get_settings  # noqa: PLC0415
    cfg = get_settings().store
    backend = cfg.graph_backend

    if backend == "sqlite":
        from cce.index.db import get_db  # noqa: PLC0415
        from cce.graph.sqlite_store import SQLiteGraphStore  # noqa: PLC0415
        return SQLiteGraphStore(get_db())

    if backend == "duckdb":
        from cce.graph.duckdb_store import DuckDBGraphStore  # noqa: PLC0415
        return DuckDBGraphStore(dsn=cfg.graph_dsn)

    if backend == "postgres":
        from cce.graph.postgres_store import PostgresGraphStore  # noqa: PLC0415
        return PostgresGraphStore(dsn=cfg.graph_dsn)

    raise ValueError(
        f"Unknown graph backend: {backend!r}. "
        "Set CCE_STORE__GRAPH_BACKEND to sqlite | duckdb | postgres."
    )
