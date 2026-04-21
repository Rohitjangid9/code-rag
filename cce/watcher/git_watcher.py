"""Phase 10 — Git-based change detection for CI/CD incremental re-indexing.

Used when you want to re-index only the files that changed in the last commit
(or between two refs) rather than running a full file-system watch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from cce.logging import get_logger

log = get_logger(__name__)

_SOURCE_EXTS = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})


def changed_files_since_commit(
    root: Path,
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
) -> tuple[list[Path], list[str]]:
    """Return (modified_paths, deleted_rel_paths) between two git refs.

    *modified_paths* are absolute Paths that exist on disk (new/changed).
    *deleted_rel_paths* are repo-relative strings for files that were removed.
    """
    raw = _git_diff_names(root, base_ref, head_ref)
    modified, deleted = [], []
    for rel in raw:
        if not any(rel.endswith(ext) for ext in _SOURCE_EXTS):
            continue
        abs_path = root / rel
        if abs_path.exists():
            modified.append(abs_path)
        else:
            deleted.append(rel)
    return modified, deleted


def changed_files_unstaged(root: Path) -> list[Path]:
    """Return absolute paths of source files modified but not yet committed."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
        paths = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if any(line.endswith(ext) for ext in _SOURCE_EXTS):
                abs_path = root / line
                if abs_path.exists():
                    paths.append(abs_path)
        return paths
    except Exception as exc:  # noqa: BLE001
        log.debug("git diff failed: %s", exc)
        return []


def current_commit(root: Path) -> str | None:
    """Return the current HEAD commit SHA, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def reindex_git_diff(pipeline, root: Path, base_ref: str = "HEAD~1") -> dict:
    """Re-index only files that changed since *base_ref*. Returns stats dict."""
    modified, deleted = changed_files_since_commit(root, base_ref=base_ref)

    from cce.walker import WalkedFile, _EXT_LANG  # noqa: PLC0415
    from cce.hashing import delete_file_records  # noqa: PLC0415

    stats = {"reindexed": 0, "deleted": 0, "errors": []}

    for abs_path in modified:
        try:
            lang = _EXT_LANG.get(abs_path.suffix.lower())
            if not lang:
                continue
            wf = WalkedFile(path=abs_path, language=lang,
                            rel_path=abs_path.relative_to(root))
            dummy_stats = type("S", (), {"symbols_indexed": 0, "edges_indexed": 0, "errors": []})()
            pipeline._index_file(wf, root, ["lexical", "symbols", "graph", "framework"],
                                  dummy_stats, active_frameworks=set())
            stats["reindexed"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"{abs_path}: {exc}")

    for rel in deleted:
        try:
            delete_file_records(rel, pipeline.symbol_store._db)
            stats["deleted"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"{rel}: {exc}")

    return stats


def _git_diff_names(root: Path, base_ref: str, head_ref: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, head_ref],
            cwd=str(root), capture_output=True, text=True, timeout=15,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        log.warning("git diff failed: %s", exc)
        return []
