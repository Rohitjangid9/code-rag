"""Phase 3 — Symbol extraction and symbol store tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cce.graph.schema import Language, NodeKind
from cce.parsers.tree_sitter_parser import TreeSitterParser
from cce.index.db import get_db
from cce.index.symbol_store import SymbolStore

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PY = FIXTURES / "sample_python"


@pytest.fixture
def parser():
    return TreeSitterParser()


@pytest.fixture
def sym_store(tmp_path):
    db = get_db(tmp_path / "sym_test.sqlite")
    return SymbolStore(db)


def _parse_file(parser, path: Path, root: Path, lang: Language):
    source = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    return parser.parse(path, rel, lang, source)


# ── Python parsing ─────────────────────────────────────────────────────────────

def test_extracts_python_classes(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    kinds = {n.kind for n in pf.nodes}
    assert NodeKind.CLASS in kinds
    names = {n.name for n in pf.nodes}
    assert "User" in names
    assert "AdminUser" in names


def test_extracts_python_methods(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    methods = [n for n in pf.nodes if n.kind == NodeKind.METHOD]
    method_names = {m.name for m in methods}
    assert "full_name" in method_names
    assert "is_admin" in method_names


def test_extracts_python_functions(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    funcs = [n for n in pf.nodes if n.kind == NodeKind.FUNCTION]
    func_names = {f.name for f in funcs}
    assert "get_user_by_email" in func_names
    assert "create_user" in func_names


def test_python_docstring_captured(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    user_class = next((n for n in pf.nodes if n.name == "User"), None)
    assert user_class is not None
    assert user_class.docstring and "user" in user_class.docstring.lower()


def test_python_line_numbers(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    user = next(n for n in pf.nodes if n.name == "User")
    assert user.line_start >= 1
    assert user.line_end > user.line_start


def test_python_inherits_edge(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    from cce.graph.schema import EdgeKind  # noqa: PLC0415
    inherits = [e for e in pf.raw_edges if e.kind == EdgeKind.INHERITS]
    dst_names = {e.dst_qualified_name for e in inherits}
    assert "models.User" in dst_names   # AdminUser inherits models.User


def test_qualified_name_structure(parser):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    user = next(n for n in pf.nodes if n.name == "User" and n.kind == NodeKind.CLASS)
    assert "models" in user.qualified_name
    assert "User" in user.qualified_name

    full_name_method = next((n for n in pf.nodes if n.name == "full_name"), None)
    assert full_name_method is not None
    assert "User" in full_name_method.qualified_name


# ── TSX parsing ────────────────────────────────────────────────────────────────

def test_extracts_react_component(parser):
    tsx_path = FIXTURES / "sample_react" / "UserCard.tsx"
    pf = _parse_file(parser, tsx_path, FIXTURES / "sample_react", Language.TSX)
    components = [n for n in pf.nodes if n.kind == NodeKind.COMPONENT]
    assert any(c.name == "UserCard" for c in components)


def test_extracts_tsx_hook_usage_edge(parser):
    from cce.graph.schema import EdgeKind  # noqa: PLC0415
    tsx_path = FIXTURES / "sample_react" / "UserCard.tsx"
    pf = _parse_file(parser, tsx_path, FIXTURES / "sample_react", Language.TSX)
    # imports generate IMPORTS raw edges
    import_edges = [e for e in pf.raw_edges if e.kind == EdgeKind.IMPORTS]
    assert len(import_edges) >= 1


# ── Symbol store ───────────────────────────────────────────────────────────────

def test_upsert_and_retrieve(parser, sym_store):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    sym_store.upsert_many(pf.nodes)

    user = sym_store.get_by_qname(next(n.qualified_name for n in pf.nodes if n.name == "User"))
    assert user is not None
    assert user.name == "User"


def test_delete_for_file(parser, sym_store):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    sym_store.upsert_many(pf.nodes)
    sym_store.delete_for_file(pf.rel_path)
    assert sym_store.get_for_file(pf.rel_path) == []


def test_fts_search(parser, sym_store):
    pf = _parse_file(parser, SAMPLE_PY / "models.py", SAMPLE_PY, Language.PYTHON)
    sym_store.upsert_many(pf.nodes)
    hits = sym_store.search("AdminUser")
    assert len(hits) >= 1
    assert any(h.node.name == "AdminUser" for h in hits)
