"""Detect which frameworks a repo uses by inspecting its structure + contents.

F-M10: detection now prefers AST-level checks over substring matches and
accepts a pre-walked list of files so we never re-walk ``node_modules`` or
``.venv`` (which ``rglob`` would happily descend into).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

from cce.graph.schema import FrameworkTag


# Files whose AST is too large / exotic to parse quickly
_MAX_AST_BYTES = 500_000


def _imports_module(tree: ast.Module, prefixes: tuple[str, ...]) -> bool:
    """Return True if *tree* has ``import X`` or ``from X …`` where X matches."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod == p or mod.startswith(p + ".") for p in prefixes):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == p or alias.name.startswith(p + ".") for p in prefixes):
                    return True
    return False


def _has_django_urlpatterns(tree: ast.Module) -> bool:
    """True when the module binds a top-level name ``urlpatterns``."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "urlpatterns":
                    return True
    return False


def _safe_parse(py: Path) -> ast.Module | None:
    """Parse a Python file to AST or return None on any failure / size cap."""
    try:
        if py.stat().st_size > _MAX_AST_BYTES:
            return None
        return ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError, ValueError):
        return None


def detect_frameworks(
    root: Path,
    python_files: Iterable[Path] | None = None,
) -> set[FrameworkTag]:
    """Return the set of frameworks present in *root*.

    When *python_files* is provided the walk is skipped — callers (like
    :class:`IndexPipeline`) already have a gitignore-aware file list and
    re-using it avoids descending into ``node_modules`` / ``.venv``.
    """
    found: set[FrameworkTag] = set()

    # --- Django (marker file is the cheapest check) ---
    if (root / "manage.py").exists():
        found.add(FrameworkTag.DJANGO)

    # --- Collect Python files (fallback to rglob only if caller didn't give us any) ---
    if python_files is None:
        python_files = [p for p in root.rglob("*.py") if p.stat().st_size < _MAX_AST_BYTES]
    else:
        python_files = list(python_files)

    for py in python_files:
        # Short-circuit once we've found every Python-side framework
        if {FrameworkTag.FASTAPI, FrameworkTag.DRF, FrameworkTag.DJANGO} <= found:
            break
        tree = _safe_parse(py)
        if tree is None:
            continue
        if FrameworkTag.FASTAPI not in found and _imports_module(tree, ("fastapi",)):
            found.add(FrameworkTag.FASTAPI)
        if FrameworkTag.DRF not in found and _imports_module(tree, ("rest_framework",)):
            found.add(FrameworkTag.DRF)
        if FrameworkTag.DJANGO not in found:
            if _imports_module(tree, ("django",)) or (
                py.name == "urls.py" and _has_django_urlpatterns(tree)
            ):
                found.add(FrameworkTag.DJANGO)

    # --- React (package.json declarations, then .tsx/.jsx file presence) ---
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "react" in all_deps:
                found.add(FrameworkTag.REACT)
        except Exception:  # noqa: BLE001
            pass
    if FrameworkTag.REACT not in found:
        if any(root.rglob("*.tsx")) or any(root.rglob("*.jsx")):
            found.add(FrameworkTag.REACT)

    return found


def file_belongs_to(path: Path, source: str, frameworks: set[FrameworkTag]) -> set[FrameworkTag]:
    """Return which frameworks are relevant for a *single file*."""
    relevant: set[FrameworkTag] = set()
    name = path.name.lower()
    src_lower = source[:4000]  # only scan header for speed

    if FrameworkTag.DJANGO in frameworks or FrameworkTag.DRF in frameworks:
        django_signals = (
            "urlpatterns" in source
            or "models.Model" in source
            or "ModelSerializer" in source
            or "Serializer" in source
            or "MIDDLEWARE" in source
            or "@receiver" in source
            or "admin.site.register" in source
        )
        if django_signals:
            if "urlpatterns" in source:
                relevant.add(FrameworkTag.DJANGO)
            if "models.Model" in source:
                relevant.add(FrameworkTag.DJANGO)
            if "ModelSerializer" in source or "Serializer" in source:
                relevant.add(FrameworkTag.DRF)
            if "MIDDLEWARE" in source and name == "settings.py":
                relevant.add(FrameworkTag.DJANGO)
            if "@receiver" in source:
                relevant.add(FrameworkTag.DJANGO)

    if FrameworkTag.FASTAPI in frameworks:
        if "FastAPI" in source or "APIRouter" in source or "@app." in source or "@router." in source:
            relevant.add(FrameworkTag.FASTAPI)

    if FrameworkTag.REACT in frameworks:
        if path.suffix in (".tsx", ".jsx") or "createBrowserRouter" in source or "<Route" in source:
            relevant.add(FrameworkTag.REACT)

    return relevant
