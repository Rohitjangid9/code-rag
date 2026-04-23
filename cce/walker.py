"""Gitignore-aware recursive file walker with language detection.

F29: Adds Go, Java, and Rust to the extension-language map.
F35: Loads ``.cceignore`` from the repo root (same gitwildmatch syntax as
     ``.gitignore``) and merges it with the gitignore spec so project teams
     can exclude generated or vendor files without touching ``.gitignore``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from cce.graph.schema import Language

# F-M6: two-tier skip list.
#
#   _HARD_SKIP — never useful to index (VCS metadata, caches, dep installs,
#                build artefacts).  Cannot be overridden.
#   _SOFT_SKIP — skipped by default, but opt-in-able via ``include_dirs``
#                or the CLI ``--include`` flag.  These contain real code
#                that some queries legitimately need (Django migrations,
#                Go vendor forks, Rust/Maven build output when probing
#                generated sources).
_HARD_SKIP: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", ".pnp",
    "dist", "build", "out", ".next", ".nuxt", ".output",
    "coverage", ".coverage",
    "eggs", ".eggs", "*.egg-info",
    ".tox", ".nox",
})

_SOFT_SKIP: frozenset[str] = frozenset({
    "migrations",          # Django migrations
    "vendor",              # Go vendor directories
    "target",              # Rust/Maven build output
})

# Backward-compat alias — external code (tests, plugins) may still import it.
_ALWAYS_SKIP: frozenset[str] = _HARD_SKIP | _SOFT_SKIP

# File extensions → Language (F29: Go, Java, Rust added)
_EXT_LANG: dict[str, Language] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JAVASCRIPT,
    ".jsx":  Language.JSX,
    ".ts":   Language.TYPESCRIPT,
    ".tsx":  Language.TSX,
    ".go":   Language.GO,
    ".java": Language.JAVA,
    ".rs":   Language.RUST,
}

# Max file size to index (skip generated / minified blobs)
_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


class WalkedFile:
    __slots__ = ("path", "language", "rel_path")

    def __init__(self, path: Path, language: Language, rel_path: Path) -> None:
        self.path = path
        self.language = language
        self.rel_path = rel_path


def _load_pathspec(root: Path, filename: str):
    """Return a pathspec matcher for *filename* in *root*, or None."""
    try:
        import pathspec  # noqa: PLC0415
    except ImportError:
        return None
    p = root / filename
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _load_gitignore(root: Path):
    """Return a pathspec matcher for the repo root .gitignore (if any)."""
    return _load_pathspec(root, ".gitignore")


def _load_cceignore(root: Path):
    """Return a pathspec matcher for .cceignore (F35), or None if absent."""
    return _load_pathspec(root, ".cceignore")


def _is_ignored(rel: Path, *specs) -> bool:
    """Return True if *rel* matches any of the given pathspec matchers."""
    rel_str = str(rel)
    return any(s is not None and s.match_file(rel_str) for s in specs)


def walk_repo(
    root: Path,
    skip_dirs: set[str] | None = None,
    include_dirs: set[str] | None = None,
) -> Iterator[WalkedFile]:
    """Yield WalkedFile for every indexable source file under *root*.

    Respects the repo's root ``.gitignore`` and ``.cceignore`` (F35) and
    skips known junk directories.

    F-M6 parameters:
        skip_dirs:    extra directory names to skip beyond the built-in set.
        include_dirs: directory names to *un-skip* from ``_SOFT_SKIP``
                      (e.g. ``{"migrations"}`` to index Django migrations).
    """
    root = root.resolve()
    gitignore = _load_gitignore(root)
    cceignore = _load_cceignore(root)   # F35
    effective_soft = _SOFT_SKIP - (include_dirs or set())
    skips = _HARD_SKIP | effective_soft | (skip_dirs or set())

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

            # Check gitignore and .cceignore (F35)
            if _is_ignored(rel, gitignore, cceignore):
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
