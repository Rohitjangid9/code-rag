"""Phase 10 — File watcher and git change-detection tests."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── File watcher: debounce + event dispatch ────────────────────────────────────

def test_schedule_and_flush(tmp_path):
    from cce.watcher.file_watcher import CodeChangeHandler  # noqa: PLC0415

    pipeline = MagicMock()
    handler = CodeChangeHandler(pipeline, tmp_path, debounce_s=0.05)

    py_file = str(tmp_path / "app.py")
    handler._schedule(py_file, "upsert")
    assert len(handler._pending) == 1

    time.sleep(0.1)
    processed = handler.flush()
    assert processed == 1
    assert len(handler._pending) == 0


def test_non_source_file_ignored(tmp_path):
    from cce.watcher.file_watcher import CodeChangeHandler  # noqa: PLC0415

    handler = CodeChangeHandler(MagicMock(), tmp_path, debounce_s=0.0)
    handler._schedule(str(tmp_path / "README.md"), "upsert")
    assert len(handler._pending) == 0


def test_debounce_prevents_early_flush(tmp_path):
    from cce.watcher.file_watcher import CodeChangeHandler  # noqa: PLC0415

    handler = CodeChangeHandler(MagicMock(), tmp_path, debounce_s=5.0)
    handler._schedule(str(tmp_path / "views.py"), "upsert")
    # Should not flush yet (deadline 5s in future)
    processed = handler.flush()
    assert processed == 0


def test_delete_action_calls_delete_records(tmp_path):
    from cce.watcher.file_watcher import CodeChangeHandler  # noqa: PLC0415

    pipeline = MagicMock()
    handler = CodeChangeHandler(pipeline, tmp_path, debounce_s=0.0)
    # Create a real .py file so _schedule passes ext check
    py_file = tmp_path / "models.py"
    py_file.touch()

    handler._schedule(str(py_file), "delete")
    py_file.unlink()  # simulate deletion before flush

    with patch("cce.watcher.file_watcher.delete_file_records") as mock_del:
        handler.flush()
        mock_del.assert_called_once()


def test_file_watcher_start_stop(tmp_path):
    from cce.watcher.file_watcher import FileWatcher  # noqa: PLC0415

    pipeline = MagicMock()
    pipeline.symbol_store._db = MagicMock()

    watcher = FileWatcher(pipeline, tmp_path, debounce_s=0.5, poll_interval=0.1)
    watcher.start()
    assert watcher._observer.is_alive()
    watcher.stop()
    assert not watcher._observer.is_alive()


def test_context_manager(tmp_path):
    from cce.watcher.file_watcher import FileWatcher  # noqa: PLC0415

    pipeline = MagicMock()
    with FileWatcher(pipeline, tmp_path, debounce_s=0.5):
        pass  # should not raise


# ── Git watcher ────────────────────────────────────────────────────────────────

def test_changed_files_parses_diff_output(tmp_path):
    from cce.watcher.git_watcher import _git_diff_names  # noqa: PLC0415

    fake_output = "app/views.py\napp/models.py\nREADME.md\n"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=fake_output, returncode=0)
        result = _git_diff_names(tmp_path, "HEAD~1", "HEAD")
    assert "app/views.py" in result
    assert "app/models.py" in result
    assert "README.md" in result


def test_changed_files_since_commit_filters_extensions(tmp_path):
    from cce.watcher.git_watcher import changed_files_since_commit  # noqa: PLC0415

    # Create real files so exists() passes
    (tmp_path / "views.py").write_text("x=1")
    (tmp_path / "styles.css").write_text("body{}")

    with patch("cce.watcher.git_watcher._git_diff_names",
               return_value=["views.py", "styles.css", "deleted_model.py"]):
        modified, deleted = changed_files_since_commit(tmp_path)

    # views.py exists → modified; deleted_model.py missing → deleted; styles.css → filtered
    assert any("views.py" in str(p) for p in modified)
    assert "deleted_model.py" in deleted
    assert not any("styles.css" in str(p) for p in modified)


def test_current_commit_returns_string_or_none(tmp_path):
    from cce.watcher.git_watcher import current_commit  # noqa: PLC0415

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="abc123def456\n", returncode=0)
        sha = current_commit(tmp_path)
    assert sha == "abc123def456"


def test_git_diff_names_handles_failure(tmp_path):
    from cce.watcher.git_watcher import _git_diff_names  # noqa: PLC0415

    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        result = _git_diff_names(tmp_path, "HEAD~1", "HEAD")
    assert result == []
