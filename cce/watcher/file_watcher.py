"""Phase 10 — Watchdog-based file watcher for incremental re-indexing.

Debounces rapid saves (1 s default), re-indexes on modify/create,
removes stale records on delete. Runs in a daemon thread.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from cce.logging import get_logger

log = get_logger(__name__)

_SOURCE_EXTS = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})


class CodeChangeHandler(FileSystemEventHandler):
    """Debounced handler: buffers events and flushes after a quiet period."""

    def __init__(self, pipeline, root: Path, debounce_s: float = 1.0) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._root = root
        self._debounce_s = debounce_s
        self._pending: dict[str, tuple[str, float]] = {}  # path → (event_type, deadline)
        self._lock = threading.Lock()

    # ── watchdog callbacks ────────────────────────────────────────────────────

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "upsert")

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "upsert")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "delete")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(getattr(event, "src_path", ""), "delete")
            self._schedule(getattr(event, "dest_path", ""), "upsert")

    # ── debounce flush ────────────────────────────────────────────────────────

    def _schedule(self, abs_path: str, action: str) -> None:
        if Path(abs_path).suffix not in _SOURCE_EXTS:
            return
        with self._lock:
            self._pending[abs_path] = (action, time.monotonic() + self._debounce_s)

    def flush(self) -> int:
        """Flush all events past their debounce deadline. Returns number processed."""
        now = time.monotonic()
        ready: list[tuple[str, str]] = []
        with self._lock:
            for path, (action, deadline) in list(self._pending.items()):
                if now >= deadline:
                    ready.append((path, action))
                    del self._pending[path]

        for abs_path, action in ready:
            try:
                self._process(abs_path, action)
            except Exception as exc:  # noqa: BLE001
                log.warning("Watcher error for %s: %s", abs_path, exc)
        return len(ready)

    def _process(self, abs_path: str, action: str) -> None:
        path = Path(abs_path)
        try:
            rel = str(path.relative_to(self._root))
        except ValueError:
            return

        if action == "delete":
            from cce.hashing import delete_file_records  # noqa: PLC0415
            delete_file_records(rel, self._pipeline.symbol_store._db)
            log.info("Watcher: deleted %s", rel)
        else:
            if not path.exists() or path.stat().st_size > 1_000_000:
                return
            from cce.walker import WalkedFile, _EXT_LANG  # noqa: PLC0415
            lang = _EXT_LANG.get(path.suffix.lower())
            if not lang:
                return
            wf = WalkedFile(path=path, language=lang, rel_path=path.relative_to(self._root))
            stats = self._pipeline._index_file.__func__  # bound method
            self._pipeline._index_file(
                wf, self._root,
                layers=["lexical", "symbols", "graph", "framework"],
                stats=type("S", (), {"symbols_indexed": 0, "edges_indexed": 0, "errors": []})(),
                active_frameworks=set(),
            )
            log.info("Watcher: re-indexed %s", rel)


class FileWatcher:
    """Manages the watchdog Observer + flush loop."""

    def __init__(self, pipeline, root: Path, debounce_s: float = 1.0,
                 poll_interval: float = 0.5) -> None:
        self._handler = CodeChangeHandler(pipeline, root, debounce_s)
        self._root = root
        self._poll = poll_interval
        self._observer = Observer()
        self._observer.schedule(self._handler, str(root), recursive=True)
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._observer.start()
        self._stop_event.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        log.info("Watcher started on %s", self._root)

    def stop(self) -> None:
        self._stop_event.set()
        self._observer.stop()
        self._observer.join(timeout=5)
        log.info("Watcher stopped")

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._handler.flush()
            time.sleep(self._poll)

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
