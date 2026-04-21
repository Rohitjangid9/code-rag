"""Gitignore-aware recursive file walker with language detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from cce.graph.schema import Language

# Directories that are always skipped regardless of .gitignore
_ALWAYS_SKIP: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", ".pnp",
    "dist", "build", "out", ".next", ".nuxt", ".output",
    "coverage", ".coverage",
    "eggs", ".eggs", "*.egg-info",
    ".tox", ".nox",
    "migrations",          # keep code; skip for default (user can override)
})

# File extensions → Language
_EXT_LANG: dict[str, Language] = {
    ".py":  Language.PYTHON,
    ".js":  Language.JAVASCRIPT,
    ".jsx": Language.JSX,
    ".ts":  Language.TYPESCRIPT,
    ".tsx": Language.TSX,
}

# Max file size to index (skip generated / minified blobs)
_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


class WalkedFile:
    __slots__ = ("path", "language", "rel_path")

    def __init__(self, path: Path, language: Language, rel_path: Path) -> None:
        self.path = path
        self.language = language
        self.rel_path = rel_path


def _load_gitignore(root: Path):
    """Return a pathspec matcher for the repo root .gitignore (if any)."""
    try:
        import pathspec  # noqa: PLC0415
    except ImportError:
        return None

    gi_path = root / ".gitignore"
    if not gi_path.exists():
        return None
    lines = gi_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def walk_repo(root: Path, skip_dirs: set[str] | None = None) -> Iterator[WalkedFile]:
    """Yield WalkedFile for every indexable source file under *root*.

    Respects the repo's root .gitignore and skips known junk directories.
    """
    root = root.resolve()
    gitignore = _load_gitignore(root)
    skips = _ALWAYS_SKIP | (skip_dirs or set())

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        dirpath = Path(dirpath_str)

        # Prune dirs in-place so os.walk doesn't recurse into them
        dirnames[:] = [
            d for d in dirnames
            if d not in skips
            and not d.startswith(".")
            or d in (".github",)  # allow .github but block rest of dotdirs
        ]
        # Also prune dotdirs (already handled above but be explicit)
        dirnames[:] = [d for d in dirnames if not (d.startswith(".") and d != ".github")]

        for fname in filenames:
            fpath = dirpath / fname
            ext = fpath.suffix.lower()
            if ext not in _EXT_LANG:
                continue

            rel = fpath.relative_to(root)

            # Check gitignore
            if gitignore and gitignore.match_file(str(rel)):
                continue

            # Skip huge files (minified / generated)
            try:
                if fpath.stat().st_size > _MAX_BYTES:
                    continue
            except OSError:
                continue

            yield WalkedFile(path=fpath, language=_EXT_LANG[ext], rel_path=rel)


def file_to_module_qname(file_path: Path, root: Path) -> str:
    """Convert a file path to a dotted module name relative to *root*.

    Example: root=``/repo`` path=``/repo/app/views.py`` → ``app.views``
    """
    try:
        rel = file_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = file_path
    return ".".join(rel.with_suffix("").parts)
