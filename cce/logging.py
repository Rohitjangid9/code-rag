"""Logging setup using rich for console output.

Configurable via .env / environment variables:

    CCE_LOG_LEVEL=DEBUG              # root log level (DEBUG|INFO|WARNING|ERROR)
    CCE_INDEXER__VERBOSE=true        # per-file summary lines during indexing
    CCE_INDEXER__LOG_FILE=.cce/indexer.log  # optional dedicated log file
    CCE_INDEXER__JEDI_DEBUG=true     # detailed Jedi resolution traces
    CCE_INDEXER__EDGE_DEBUG=true     # per-file edge-kind breakdown
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from rich.logging import RichHandler

from cce.config import get_settings

_configured = False
_file_handler: logging.FileHandler | None = None

# Logger names that belong to the indexing pipeline.
# When jedi_debug / edge_debug are enabled these get set to DEBUG
# even when the root logger is at INFO.
_INDEXER_LOGGERS = (
    "cce.indexer",
    "cce.parsers.python_resolver",
    "cce.parsers.js_resolver",
    "cce.parsers.tree_sitter_parser",
    "cce.graph.sqlite_store",
    "cce.index.symbol_store",
    "cce.index.lexical_store",
    "cce.extractors",
)


def setup_logging() -> None:
    """Configure root logger once with a Rich console handler.

    Call this once at startup (e.g. in the CLI entry point).  Subsequent
    calls are no-ops unless :func:`reset_logging` has been called first.
    """
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

    # ── Optional file handler ──────────────────────────────────────────────
    if settings.indexer.log_file:
        _attach_file_handler(settings.indexer.log_file, settings.log_level)

    # ── Sub-logger level overrides for indexing diagnostics ───────────────
    _apply_indexer_log_levels(settings)

    _configured = True


def _attach_file_handler(log_file: Path, root_level: str) -> None:
    """Add a rotating file handler that captures all indexer logs."""
    global _file_handler

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    _file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.DEBUG)  # capture everything; filter per logger

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file_handler.setFormatter(fmt)
    logging.getLogger().addHandler(_file_handler)


def _apply_indexer_log_levels(settings) -> None:
    """Lower specific logger levels when diagnostic flags are on."""
    idx = settings.indexer
    # Any diagnostic flag drops the indexer loggers to DEBUG
    if idx.verbose or idx.jedi_debug or idx.edge_debug:
        for name in _INDEXER_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)


def attach_indexer_file_handler(log_file: Path) -> None:
    """Attach (or re-attach) a file handler at runtime.

    Useful when the pipeline is invoked programmatically with a custom path
    that differs from the .env default.
    """
    settings = get_settings()
    _attach_file_handler(log_file, settings.log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the root handler if needed."""
    setup_logging()
    return logging.getLogger(name)


def reset_logging() -> None:
    """Reset logging state — intended for tests only."""
    global _configured, _file_handler
    if _file_handler:
        logging.getLogger().removeHandler(_file_handler)
        _file_handler = None
    _configured = False
