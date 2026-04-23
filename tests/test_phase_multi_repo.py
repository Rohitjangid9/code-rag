"""F-M1..F-M5: multi-repo isolation regression tests.

These tests guard against the ``lru_cache(maxsize=1)`` regression where
indexing repo A and then repo B would leave all queries pinned to A.
"""

from __future__ import annotations

import os

import pytest


# ── F-M1: PathsSettings.resolve + get_settings(repo_root=...) ────────────────

def test_paths_resolve_anchors_relative_paths(tmp_path):
    from cce.config import PathsSettings  # noqa: PLC0415

    p = PathsSettings()  # defaults are relative (".cce/index.sqlite", …)
    resolved = p.resolve(tmp_path)

    assert resolved.data_dir == tmp_path.resolve() / ".cce"
    assert resolved.sqlite_db == tmp_path.resolve() / ".cce" / "index.sqlite"
    assert resolved.qdrant_path == tmp_path.resolve() / ".cce" / "qdrant"


def test_paths_resolve_leaves_absolute_paths_alone(tmp_path):
    from cce.config import PathsSettings  # noqa: PLC0415

    absolute = tmp_path / "custom.sqlite"
    p = PathsSettings(sqlite_db=absolute)
    resolved = p.resolve(tmp_path / "someroot")

    assert resolved.sqlite_db == absolute  # absolute override preserved


def test_get_settings_is_repo_keyed(tmp_path):
    from cce.config import get_settings, reset_settings_cache  # noqa: PLC0415

    reset_settings_cache()
    try:
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()

        s_a = get_settings(repo_root=repo_a)
        s_b = get_settings(repo_root=repo_b)

        assert s_a is not s_b
        assert s_a.repo_root == repo_a.resolve()
        assert s_b.repo_root == repo_b.resolve()
        assert s_a.paths.sqlite_db != s_b.paths.sqlite_db
        assert s_a.paths.sqlite_db.is_relative_to(repo_a.resolve())
        assert s_b.paths.sqlite_db.is_relative_to(repo_b.resolve())
    finally:
        reset_settings_cache()


def test_walk_up_finds_cce_marker(tmp_path):
    from cce.config import _walk_up_find_cce  # noqa: PLC0415

    (tmp_path / ".cce").mkdir()
    (tmp_path / ".cce" / "index.json").write_text("{}")
    nested = tmp_path / "src" / "deep" / "sub"
    nested.mkdir(parents=True)

    found = _walk_up_find_cce(nested)
    assert found == tmp_path.resolve()


def test_walk_up_returns_none_when_no_marker(tmp_path):
    from cce.config import _walk_up_find_cce  # noqa: PLC0415

    assert _walk_up_find_cce(tmp_path) is None


def test_env_var_overrides_walk_up(tmp_path, monkeypatch):
    from cce.config import get_settings, reset_settings_cache  # noqa: PLC0415

    reset_settings_cache()
    monkeypatch.setenv("CCE_REPO_ROOT", str(tmp_path))
    try:
        s = get_settings()
        assert s.repo_root == tmp_path.resolve()
    finally:
        reset_settings_cache()


# ── F-M2: retrieval tools cache is repo-keyed ─────────────────────────────────

def test_pipeline_cache_is_repo_keyed(tmp_path):
    from cce.config import reset_settings_cache  # noqa: PLC0415
    from cce.retrieval.tools import _pipeline, reset_retrieval_cache  # noqa: PLC0415

    reset_settings_cache()
    reset_retrieval_cache()
    try:
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()

        p_a = _pipeline(repo_root=repo_a)
        p_b = _pipeline(repo_root=repo_b)

        assert p_a is not p_b
        assert p_a is _pipeline(repo_root=repo_a)  # stable within a repo
    finally:
        reset_settings_cache()
        reset_retrieval_cache()


# ── F-M3: collection name is consistent between indexer and retriever ────────

def test_collection_name_is_stable_per_root(tmp_path):
    pytest.importorskip("qdrant_client")
    from cce.config import Settings  # noqa: PLC0415
    from cce.index.vector_store import VectorStore  # noqa: PLC0415

    s = Settings()
    s.paths.qdrant_path = tmp_path / "qdrant"
    s.paths.data_dir = tmp_path
    vs = VectorStore(s)

    assert vs.collection_name(tmp_path) == vs.collection_name(tmp_path.resolve())
    assert vs.collection_name(tmp_path) != vs.collection_name(tmp_path / "sub")


# ── F-M4/F-M5: end-to-end isolation across two repos from a third CWD ────────

def test_two_repos_indexed_from_third_cwd_stay_isolated(tmp_path, monkeypatch):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "alpha.py").write_text("def only_in_a():\n    return 1\n")
    (repo_b / "beta.py").write_text("def only_in_b():\n    return 2\n")

    # CWD is a third, unrelated directory — exercises the "paths relative to
    # repo root not CWD" contract from F-M1.
    third = tmp_path / "third"
    third.mkdir()
    monkeypatch.chdir(third)

    def _make_settings(root):
        s = Settings()
        s.paths = s.paths.resolve(root)
        s.repo_root = root.resolve()
        return s

    IndexPipeline(settings=_make_settings(repo_a)).run(repo_a, layers=["lexical", "symbols"])
    IndexPipeline(settings=_make_settings(repo_b)).run(repo_b, layers=["lexical", "symbols"])

    qnames_a = set(IndexPipeline(settings=_make_settings(repo_a)).symbol_store.list_qnames())
    qnames_b = set(IndexPipeline(settings=_make_settings(repo_b)).symbol_store.list_qnames())

    assert any(q.endswith("only_in_a") for q in qnames_a)
    assert any(q.endswith("only_in_b") for q in qnames_b)
    assert not any(q.endswith("only_in_a") for q in qnames_b)
    assert not any(q.endswith("only_in_b") for q in qnames_a)
    # Sanity: .cce was created inside each repo, not in `third`
    assert (repo_a / ".cce" / "index.sqlite").exists()
    assert (repo_b / ".cce" / "index.sqlite").exists()
    assert not (third / ".cce").exists()
