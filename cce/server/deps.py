"""FastAPI dependency-injection helpers (shared across routes)."""

from __future__ import annotations

from cce.config import Settings, get_settings


def settings_dep() -> Settings:
    """Resolve Settings for the active repo (F-M2).

    ``get_settings()`` is itself repo-keyed now, so this wrapper no longer
    caches — each request re-resolves against ``CCE_REPO_ROOT`` / walk-up.
    """
    return get_settings()
