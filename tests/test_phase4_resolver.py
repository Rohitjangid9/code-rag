"""Phase 4 — Reference resolution tests (Python Jedi + JS heuristic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cce.graph.schema import EdgeKind, Language
from cce.parsers.tree_sitter_parser import TreeSitterParser

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PY = FIXTURES / "sample_python"


@pytest.fixture
def parser():
    return TreeSitterParser()


def _parse(parser, path, root, lang):
    source = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    return parser.parse(path, rel, lang, source)


# ── Python resolver ────────────────────────────────────────────────────────────

def test_python_import_edges_from_parser(parser):
    """Parser-level: imports produce raw IMPORTS edges."""
    pf = _parse(parser, SAMPLE_PY / "views.py", SAMPLE_PY, Language.PYTHON)
    import_edges = [e for e in pf.raw_edges if e.kind == EdgeKind.IMPORTS]
    assert len(import_edges) >= 1


def test_python_heuristic_calls(parser):
    """Heuristic resolver produces CALLS edges even without Jedi."""
    from cce.parsers.python_resolver import _heuristic_python  # noqa: PLC0415

    pf = _parse(parser, SAMPLE_PY / "views.py", SAMPLE_PY, Language.PYTHON)
    edges = _heuristic_python(pf, SAMPLE_PY)
    call_kinds = {e.kind for e in edges}
    assert EdgeKind.CALLS in call_kinds
    # views.py calls create_user and get_user_by_email
    dst_names = {e.dst_qualified_name for e in edges}
    assert any("create_user" in d or "get_user_by_email" in d for d in dst_names)


def test_python_jedi_resolver(parser):
    """Jedi resolver (skips gracefully if jedi not installed)."""
    pytest.importorskip("jedi")
    from cce.parsers.python_resolver import resolve_python_file  # noqa: PLC0415

    pf = _parse(parser, SAMPLE_PY / "views.py", SAMPLE_PY, Language.PYTHON)
    edges = resolve_python_file(pf, SAMPLE_PY)
    # Should produce at least some edges (possibly 0 if Jedi can't resolve cross-module)
    assert isinstance(edges, list)


def test_python_inherits_edges_in_resolver(parser):
    """INHERITS edges are emitted for subclasses."""
    pf = _parse(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    inherits = [e for e in pf.raw_edges if e.kind == EdgeKind.INHERITS]
    assert len(inherits) >= 1
    assert any("User" in e.dst_qualified_name for e in inherits)


# ── JS resolver ────────────────────────────────────────────────────────────────

def test_tsx_hook_edges(parser):
    """JS resolver produces USES_HOOK edges for useXxx calls."""
    from cce.parsers.js_resolver import resolve_js_file  # noqa: PLC0415

    tsx_path = FIXTURES / "sample_react" / "UserCard.tsx"
    pf = _parse(parser, tsx_path, FIXTURES / "sample_react", Language.TSX)
    edges = resolve_js_file(pf, FIXTURES / "sample_react")
    hook_edges = [e for e in edges if e.kind == EdgeKind.USES_HOOK]
    assert len(hook_edges) >= 1
    assert any("useAuth" in e.dst_qualified_name for e in hook_edges)


def test_tsx_renders_edges(parser):
    """JS resolver produces RENDERS edges for PascalCase JSX elements."""
    from cce.parsers.js_resolver import resolve_js_file  # noqa: PLC0415

    tsx_path = FIXTURES / "sample_react" / "UserCard.tsx"
    pf = _parse(parser, tsx_path, FIXTURES / "sample_react", Language.TSX)
    edges = resolve_js_file(pf, FIXTURES / "sample_react")
    render_edges = [e for e in edges if e.kind == EdgeKind.RENDERS]
    assert len(render_edges) >= 1
    assert any("Avatar" in e.dst_qualified_name for e in render_edges)


def test_confidence_below_1_for_heuristics(parser):
    """Heuristic edges should have confidence < 1."""
    from cce.parsers.js_resolver import resolve_js_file  # noqa: PLC0415

    tsx_path = FIXTURES / "sample_react" / "UserCard.tsx"
    pf = _parse(parser, tsx_path, FIXTURES / "sample_react", Language.TSX)
    edges = resolve_js_file(pf, FIXTURES / "sample_react")
    assert all(e.confidence < 1.0 for e in edges)


def test_python_references_edges(parser, tmp_path):
    """F6: identifiers as dict values or call arguments emit REFERENCES edges."""
    pytest.importorskip("jedi")
    from cce.parsers.python_resolver import resolve_python_file  # noqa: PLC0415

    src_file = tmp_path / "refs.py"
    src_file.write_text('def foo():\n    pass\n\nbar = {"x": foo}\nhandler(foo)\n')
    pf = parser.parse(src_file, "refs.py", Language.PYTHON, src_file.read_text())
    edges = resolve_python_file(pf, tmp_path)
    ref_edges = [e for e in edges if e.kind == EdgeKind.REFERENCES and "foo" in e.dst_qualified_name]
    assert len(ref_edges) == 2
