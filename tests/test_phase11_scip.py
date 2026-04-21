"""Phase 11 — SCIP export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cce.scip.schema import (
    SCIPDocument, SCIPIndex, SCIPMetadata, SCIPOccurrence,
    SCIPPosition, SCIPRelationship, SCIPSymbolInfo, SymbolRole,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sample_python"


# ── Schema dataclasses ─────────────────────────────────────────────────────────

def test_scip_position_as_list():
    pos = SCIPPosition(start_line=10, start_char=4, end_line=20, end_char=0)
    lst = pos.as_list()
    assert lst == [10, 4, 20, 0]


def test_scip_position_defaults():
    pos = SCIPPosition(start_line=5)
    lst = pos.as_list()
    assert lst[0] == 5


def test_scip_relationship_as_dict_minimal():
    rel = SCIPRelationship(symbol="scip-python python app . views.foo.", is_reference=True)
    d = rel.as_dict()
    assert d["symbol"].endswith("foo.")
    assert d["is_reference"] is True


def test_scip_relationship_empty_flags():
    rel = SCIPRelationship(symbol="scip-python python app . bar.")
    d = rel.as_dict()
    assert "is_reference" not in d
    assert "is_implementation" not in d


def test_scip_occurrence_as_dict():
    occ = SCIPOccurrence(range=[0, 0, 5, 0], symbol="scip-python python app . User.",
                         symbol_roles=SymbolRole.DEFINITION)
    d = occ.as_dict()
    assert d["range"] == [0, 0, 5, 0]
    assert d["symbol_roles"] == SymbolRole.DEFINITION


def test_scip_index_serialises_to_json():
    index = SCIPIndex()
    index.documents.append(SCIPDocument(relative_path="app/models.py", language="python"))
    raw = index.to_json()
    parsed = json.loads(raw)
    assert "documents" in parsed
    assert parsed["documents"][0]["relative_path"] == "app/models.py"
    assert parsed["metadata"]["tool_info"]["name"] == "code-context-engine"


# ── Emitter ────────────────────────────────────────────────────────────────────

def test_emitter_produces_index(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.scip.emitter import SCIPEmitter  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "scip_test.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["lexical", "symbols", "graph"])

    emitter = SCIPEmitter(pipeline.symbol_store, pipeline.graph_store)
    index = emitter.emit(root=FIXTURES)

    assert len(index.documents) >= 1
    assert any("models" in d.relative_path for d in index.documents)


def test_emitter_documents_have_occurrences(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.scip.emitter import SCIPEmitter  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "scip_occ.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["symbols"])

    emitter = SCIPEmitter(pipeline.symbol_store, pipeline.graph_store)
    index = emitter.emit(root=FIXTURES)

    for doc in index.documents:
        assert len(doc.occurrences) >= 1
        assert len(doc.symbols) >= 1


def test_emitter_symbols_have_scip_format(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.scip.emitter import SCIPEmitter  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "scip_fmt.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["symbols"])

    emitter = SCIPEmitter(pipeline.symbol_store, pipeline.graph_store)
    index = emitter.emit(root=FIXTURES)

    for doc in index.documents:
        for sym in doc.symbols:
            # SCIP symbol format: "scip-<lang> <mgr> <pkg> <version> <descriptor>"
            assert sym.symbol.startswith("scip-"), f"Bad symbol: {sym.symbol}"
            parts = sym.symbol.split()
            assert len(parts) >= 4


def test_scip_json_round_trip(tmp_path):
    from cce.config import Settings  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.scip.emitter import SCIPEmitter  # noqa: PLC0415

    settings = Settings()
    settings.paths.sqlite_db = tmp_path / "scip_rt.sqlite"
    settings.paths.data_dir = tmp_path
    settings.paths.qdrant_path = tmp_path / "qdrant"
    settings.paths.agent_checkpoint = tmp_path / "agent.sqlite"

    pipeline = IndexPipeline(settings=settings)
    pipeline.run(FIXTURES, layers=["symbols"])

    emitter = SCIPEmitter(pipeline.symbol_store, pipeline.graph_store)
    index = emitter.emit(root=FIXTURES)

    raw = index.to_json()
    parsed = json.loads(raw)

    assert "metadata" in parsed
    assert "documents" in parsed
    assert len(parsed["documents"]) >= 1
