"""Phase 2 — Lexical search tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cce.index.db import get_db
from cce.index.lexical_store import LexicalStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def lex_store(tmp_path):
    db = get_db(tmp_path / "lex_test.sqlite")
    return LexicalStore(db)


def test_upsert_and_search(lex_store: LexicalStore):
    lex_store.upsert("app/models.py", "class User:\n    def is_admin(self): pass")
    hits = lex_store.search("is_admin")
    assert len(hits) >= 1
    assert any("app/models.py" in h.path for h in hits)


def test_search_no_results(lex_store: LexicalStore):
    lex_store.upsert("app/models.py", "class User: pass")
    hits = lex_store.search("xyzzy_nonexistent_token")
    assert hits == []


def test_delete_removes_from_fts(lex_store: LexicalStore):
    lex_store.upsert("tmp/foo.py", "def authenticate_user(): pass")
    assert len(lex_store.search("authenticate_user")) >= 1
    lex_store.delete("tmp/foo.py")
    assert lex_store.search("authenticate_user") == []


def test_upsert_replaces_old_content(lex_store: LexicalStore):
    lex_store.upsert("app/views.py", "def old_function(): pass")
    lex_store.upsert("app/views.py", "def new_function(): pass")
    assert len(lex_store.search("new_function")) >= 1
    assert lex_store.search("old_function") == []


def test_search_k_limit(lex_store: LexicalStore):
    for i in range(20):
        lex_store.upsert(f"file_{i}.py", "def common_function(): pass")
    hits = lex_store.search("common_function", k=5)
    assert len(hits) <= 5


def test_index_real_fixture_files(tmp_path):
    """End-to-end: index all sample_python fixtures and search them."""
    from cce.index.db import get_db  # noqa: PLC0415
    from cce.walker import walk_repo  # noqa: PLC0415

    db = get_db(tmp_path / "e2e.sqlite")
    store = LexicalStore(db)

    sample_py = FIXTURES / "sample_python"
    for wf in walk_repo(sample_py):
        source = wf.path.read_text(encoding="utf-8", errors="ignore")
        store.upsert(str(wf.rel_path), source)

    hits = store.search("AdminUser")
    assert len(hits) >= 1

    hits2 = store.search("get_user_by_email")
    assert len(hits2) >= 1
