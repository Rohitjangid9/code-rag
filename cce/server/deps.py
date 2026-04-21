"""FastAPI dependency-injection helpers (shared across routes)."""

from __future__ import annotations

from functools import lru_cache

from cce.config import Settings, get_settings


@lru_cache(maxsize=1)
def settings_dep() -> Settings:
    """Inject the global Settings singleton."""
    return get_settings()
