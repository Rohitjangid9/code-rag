"""Phase 1 — Walker and hashing tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cce.walker import WalkedFile, file_to_module_qname, walk_repo
from cce.graph.schema import Language

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PY = FIXTURES / "sample_python"


def test_walk_finds_python_files():
    files = list(walk_repo(SAMPLE_PY))
    paths = [wf.rel_path for wf in files]
    assert any("models.py" in str(p) for p in paths)
    assert any("views.py" in str(p) for p in paths)


def test_walk_detects_language():
    files = {wf.rel_path.name: wf.language for wf in walk_repo(SAMPLE_PY)}
    assert files.get("models.py") == Language.PYTHON
    assert files.get("views.py") == Language.PYTHON


def test_walk_detects_tsx():
    sample_react = FIXTURES / "sample_react"
    files = list(walk_repo(sample_react))
    tsx_files = [wf for wf in files if wf.language == Language.TSX]
    assert len(tsx_files) >= 1
    assert any("UserCard.tsx" in str(wf.rel_path) for wf in tsx_files)


def test_walk_skips_pycache(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "foo.py").write_text("x = 1")
    (tmp_path / "real.py").write_text("x = 1")
    files = list(walk_repo(tmp_path))
    assert all("__pycache__" not in str(wf.rel_path) for wf in files)
    assert any("real.py" in str(wf.rel_path) for wf in files)


def test_file_to_module_qname():
    root = Path("/repo")
    assert file_to_module_qname(Path("/repo/app/views.py"), root) == "app.views"
    assert file_to_module_qname(Path("/repo/models.py"), root) == "models"


def test_hashing_change_detection(tmp_path):
    from cce.hashing import detect_changes, sha256_file  # noqa: PLC0415
    from cce.index.db import get_db  # noqa: PLC0415

    (tmp_path / "a.py").write_text("x = 1")
    root = tmp_path
    db = get_db(tmp_path / "test.sqlite")
    walked = list(walk_repo(root))

    cs = detect_changes(walked, db, root)
    assert len(cs.new) == 1
    assert len(cs.changed) == 0

    # Second call — nothing changed
    cs2 = detect_changes(walked, db, root)
    assert len(cs2.new) == 0
    assert len(cs2.changed) == 0

    # Mutate file — should show as changed
    (tmp_path / "a.py").write_text("x = 2")
    cs3 = detect_changes(walked, db, root)
    assert len(cs3.changed) == 1
