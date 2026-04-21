"""Logging setup using rich for console output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from cce.config import get_settings

_configured = False


def setup_logging() -> None:
    """Configure root logger once with a Rich handler."""
    global _configured
    if _configured:
        return
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring root if needed."""
    setup_logging()
    return logging.getLogger(name)
