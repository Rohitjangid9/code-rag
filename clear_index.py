"""Delete all indexed data (SQLite, Qdrant, Kùzu, agent checkpoints).

Reads paths from your .env / CCE_* env vars so custom locations are respected.
Run this before re-indexing from scratch.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _delete_path(p: Path) -> None:
    """Remove a file or directory, and SQLite journal sidecars if present."""
    if not p.exists():
        return

    if p.is_dir():
        shutil.rmtree(p)
        print(f"  removed dir  -> {p}")
    else:
        p.unlink(missing_ok=True)
        print(f"  removed file -> {p}")
        # SQLite WAL / SHM sidecars
        for suffix in ("-wal", "-shm"):
            sidecar = p.parent / (p.name + suffix)
            if sidecar.exists():
                sidecar.unlink(missing_ok=True)
                print(f"  removed file -> {sidecar}")


def main() -> None:
    # Force reload of settings in case this script is run inside a long-lived process
    import cce.config
    cce.config._settings = None

    from cce.config import get_settings

    settings = get_settings()
    paths = settings.paths

    targets = {
        "SQLite index": paths.sqlite_db,
        "Qdrant vectors": paths.qdrant_path,
        "Kùzu graph": paths.graph_db,
        "Agent checkpoint": paths.agent_checkpoint,
    }

    # Resolve everything relative to CWD so the printout is unambiguous
    resolved = {label: Path(p).resolve() for label, p in targets.items()}

    print("The following data will be deleted:")
    for label, p in resolved.items():
        status = "exists" if p.exists() else "not found"
        print(f"  [{status}] {label:<20} -> {p}")

    if not any(p.exists() for p in resolved.values()):
        print("\nNothing to delete — no index data found.")
        sys.exit(0)

    ans = input("\nType 'yes' to permanently delete the data above: ").strip().lower()
    if ans != "yes":
        print("Aborted.")
        sys.exit(0)

    print("\nDeleting ...")
    for label, p in resolved.items():
        if p.exists():
            _delete_path(p)

    # Also wipe the cached settings object so the next cce command starts clean
    cce.config._settings = None
    print("\nDone. You can now re-index with: cce index <path> [--layers ...]")


if __name__ == "__main__":
    main()
