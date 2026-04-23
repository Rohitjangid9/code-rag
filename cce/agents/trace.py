"""F13 — Structured JSONL trace log for the LangGraph agent runtime.

One JSON line per event written to ``logs/agent_trace.jsonl`` (daily rotation,
7-day retention).  Schema::

    {
        "ts":          "<ISO-8601 UTC>",
        "thread_id":   "<str>",
        "turn":        <int>,
        "node":        "planner" | "retriever" | "reasoner" | "responder",
        "tool":        "<tool name>",   // retriever events only
        "args":        {…},             // retriever events only
        "hits":        <int>,           // retriever events only
        "elapsed_ms":  <float>,
        "error":       "<str>"          // only when an error occurred
    }

Usage::

    from cce.agents.trace import emit, start_timer
    t = start_timer()
    ...
    emit({"node": "planner", "turn": 1, "thread_id": tid, "elapsed_ms": t()})
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "agent_trace.jsonl"
_lock = threading.Lock()

# Retention: keep last 7 daily log files.
_MAX_ROTATED = 7


def _rotated_path(base: Path, suffix: str) -> Path:
    return base.parent / f"{base.stem}.{suffix}{base.suffix}"


def _rotate_if_needed() -> None:
    """Rotate agent_trace.jsonl daily when the date stamp changes."""
    if not _LOG_FILE.exists():
        return
    mtime_date = datetime.fromtimestamp(_LOG_FILE.stat().st_mtime, tz=timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    if mtime_date == today:
        return
    stamp = mtime_date.isoformat()
    dest = _rotated_path(_LOG_FILE, stamp)
    try:
        _LOG_FILE.rename(dest)
    except OSError:
        return  # race-condition safe — another thread may have rotated already
    # Prune old rotated files
    rotated = sorted(_LOG_FILE.parent.glob(f"{_LOG_FILE.stem}.*.jsonl"))
    for old in rotated[:-_MAX_ROTATED]:
        try:
            old.unlink()
        except OSError:
            pass


def emit(event: dict) -> None:
    """Append one JSONL event line to the agent trace log.

    Thread-safe.  Silently swallows I/O errors so a tracing failure never
    breaks the agent.
    """
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    try:
        with _lock:
            _LOG_DIR.mkdir(exist_ok=True)
            _rotate_if_needed()
            with _LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass  # trace failure must never propagate


def start_timer() -> "callable[[], float]":
    """Return a closure that reports elapsed milliseconds since *now*."""
    t0 = time.monotonic()
    return lambda: round((time.monotonic() - t0) * 1000, 1)
