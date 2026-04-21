"""Detect which frameworks a repo uses by inspecting its structure + contents."""

from __future__ import annotations

import json
from pathlib import Path

from cce.graph.schema import FrameworkTag


def detect_frameworks(root: Path) -> set[FrameworkTag]:
    """Return the set of frameworks present in *root*.

    Checks performed (fast, no AST needed):
    - Django:  manage.py | settings.py with INSTALLED_APPS | any urls.py with urlpatterns
    - FastAPI: any .py containing 'from fastapi' or 'FastAPI()'
    - DRF:     any .py containing 'rest_framework' import
    - React:   package.json with 'react' dependency | presence of .tsx/.jsx files
    """
    found: set[FrameworkTag] = set()

    # --- Django ---
    if (root / "manage.py").exists():
        found.add(FrameworkTag.DJANGO)
    else:
        for py in root.rglob("urls.py"):
            if py.stat().st_size < 200_000:
                txt = py.read_text(encoding="utf-8", errors="ignore")
                if "urlpatterns" in txt:
                    found.add(FrameworkTag.DJANGO)
                    break

    # --- FastAPI ---
    for py in root.rglob("*.py"):
        if py.stat().st_size > 500_000:
            continue
        txt = py.read_text(encoding="utf-8", errors="ignore")
        if "from fastapi" in txt or "FastAPI()" in txt or "= FastAPI(" in txt:
            found.add(FrameworkTag.FASTAPI)
            break

    # --- DRF (mark separately so extractor runs on right files) ---
    for py in root.rglob("*.py"):
        if py.stat().st_size > 200_000:
            continue
        txt = py.read_text(encoding="utf-8", errors="ignore")
        if "rest_framework" in txt or "ModelSerializer" in txt:
            found.add(FrameworkTag.DRF)
            break

    # --- React ---
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "react" in all_deps:
                found.add(FrameworkTag.REACT)
        except Exception:  # noqa: BLE001
            pass
    if not any(t in found for t in (FrameworkTag.REACT,)):
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
