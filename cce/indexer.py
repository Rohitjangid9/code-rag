"""Phases 1-7 — IndexPipeline: walk → hash → lex → parse → framework-extract → resolve → graph → embed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from cce.config import Settings, get_settings
from cce.extractors.django_extractor import DjangoExtractor
from cce.extractors.fastapi_extractor import FastAPIExtractor
from cce.extractors.framework_detector import detect_frameworks, file_belongs_to
from cce.extractors.react_extractor import ReactExtractor
from cce.graph.schema import EdgeKind, FrameworkTag
from cce.graph.sqlite_store import SQLiteGraphStore
from cce.hashing import ChangeSet, delete_file_records, detect_changes
from cce.index.db import DatabaseManager, get_db
from cce.index.lexical_store import LexicalStore
from cce.index.symbol_store import SymbolStore
from cce.logging import get_logger
from cce.parsers.base import ParsedFile
from cce.parsers.tree_sitter_parser import TreeSitterParser
from cce.walker import WalkedFile, walk_repo

log = get_logger(__name__)

_FRAMEWORK_EXTRACTORS = [DjangoExtractor(), FastAPIExtractor(), ReactExtractor()]


@dataclass
class IndexStats:
    root: str
    files_total: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    symbols_indexed: int = 0
    edges_indexed: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


class IndexPipeline:
    """Orchestrates all indexing phases for a given codebase root."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._db: DatabaseManager | None = None
        self._lex: LexicalStore | None = None
        self._sym: SymbolStore | None = None
        self._graph: SQLiteGraphStore | None = None
        self._parser = TreeSitterParser()

    # ------------------------------------------------------------------
    # Stores — lazy init
    # ------------------------------------------------------------------

    def _init_stores(self) -> None:
        if self._db is not None:
            return
        cfg = self._settings
        self._db = get_db(cfg.paths.sqlite_db)
        self._lex = LexicalStore(self._db)
        self._sym = SymbolStore(self._db)
        self._graph = SQLiteGraphStore(self._db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, root: Path, layers: list[str] | None = None) -> IndexStats:
        """Run the full pipeline over *root*. *layers* filters which phases run."""
        layers = layers or ["lexical", "symbols", "graph", "framework"]
        t0 = time.monotonic()
        self._init_stores()

        stats = IndexStats(root=str(root))

        # Phase 1 — walk + change detection
        log.info("Walking %s …", root)
        walked: list[WalkedFile] = list(walk_repo(root))
        stats.files_total = len(walked)

        changes: ChangeSet = detect_changes(walked, self._db, root)  # type: ignore[arg-type]
        stats.files_new = len(changes.new)
        stats.files_changed = len(changes.changed)
        stats.files_deleted = len(changes.deleted)

        for rel in changes.deleted:
            delete_file_records(rel, self._db)  # type: ignore[arg-type]

        # Phase 6 — detect frameworks once for the repo
        active_frameworks: set[FrameworkTag] = set()
        if "framework" in layers:
            active_frameworks = detect_frameworks(root)
            if active_frameworks:
                log.info("Detected frameworks: %s", {f.value for f in active_frameworks})

        for wf in changes.to_index:
            try:
                self._index_file(wf, root, layers, stats, active_frameworks)
            except Exception as exc:  # noqa: BLE001
                msg = f"error indexing {wf.rel_path}: {exc}"
                log.warning(msg)
                stats.errors.append(msg)

        # Phase 7 — semantic embedding
        if "semantic" in layers:
            self._index_semantic(root, stats)

        stats.elapsed_s = time.monotonic() - t0
        log.info(
            "Done: %d new, %d changed, %d deleted, %d symbols, %d edges in %.1fs",
            stats.files_new, stats.files_changed, stats.files_deleted,
            stats.symbols_indexed, stats.edges_indexed, stats.elapsed_s,
        )
        return stats

    def _index_file(self, wf: WalkedFile, root: Path, layers: list[str],
                    stats: IndexStats, active_frameworks: set[FrameworkTag] | None = None) -> None:
        source = wf.path.read_text(encoding="utf-8", errors="replace")
        rel = str(wf.rel_path)

        # Phase 2 — lexical
        if "lexical" in layers:
            self._lex.upsert(rel, source)  # type: ignore[union-attr]

        # Phase 3 — symbol parsing (base AST)
        parsed: ParsedFile | None = None
        if "symbols" in layers or "graph" in layers or "framework" in layers:
            parsed = self._parser.parse(wf.path, rel, wf.language, source)
            self._sym.delete_for_file(rel)  # type: ignore[union-attr]
            self._sym.upsert_many(parsed.nodes)  # type: ignore[union-attr]
            stats.symbols_indexed += len(parsed.nodes)

        # Phase 6 — framework extraction (augments nodes + raw_edges)
        if "framework" in layers and parsed is not None and active_frameworks:
            relevant = file_belongs_to(wf.path, source, active_frameworks)
            for extractor in _FRAMEWORK_EXTRACTORS:
                if extractor.can_handle(wf.path, source):
                    fw_data = extractor.extract(wf.path, rel, source)
                    self._sym.upsert_many(fw_data.nodes)
                    stats.symbols_indexed += len(fw_data.nodes)
                    if parsed:
                        parsed.raw_edges.extend(fw_data.raw_edges)

        # Phase 4/5 — reference resolution → edges
        if "graph" in layers and parsed is not None:
            raw_edges = list(parsed.raw_edges)
            raw_edges += self._resolve_references(parsed, root)

            self._graph.delete_for_file(rel)  # type: ignore[union-attr]
            for re_ in raw_edges:
                dst_id = self._graph.resolve_qname(re_.dst_qualified_name)  # type: ignore[union-attr]
                if dst_id:
                    self._graph.upsert_edge(  # type: ignore[union-attr]
                        src_id=re_.src_id,
                        dst_id=dst_id,
                        kind=re_.kind,
                        file_path=re_.file_path,
                        line=re_.line,
                        confidence=re_.confidence,
                    )
                    stats.edges_indexed += 1

    # ── Phase 7 — semantic embedding ──────────────────────────────────────────

    def _index_semantic(self, root: Path, stats: IndexStats) -> None:
        """Chunk all symbols, embed with nomic-embed-code, upsert to Qdrant."""
        try:
            from cce.embeddings.chunker import chunk_nodes  # noqa: PLC0415
            from cce.embeddings.embedder import get_embedder  # noqa: PLC0415
            from cce.index.vector_store import VectorStore  # noqa: PLC0415
        except ImportError as e:
            log.warning("Semantic indexing skipped — missing dependency: %s", e)
            return

        cfg = self._settings
        embedder = get_embedder()
        vstore = VectorStore(cfg)
        collection = vstore.collection_name(root)
        vstore.ensure_collection(collection)

        # Fetch all symbols that need embedding
        conn = self._db.conn  # type: ignore[union-attr]
        rows = conn.execute("SELECT * FROM symbols").fetchall()

        from cce.index.symbol_store import _row_to_node  # noqa: PLC0415
        nodes = [_row_to_node(r) for r in rows]

        # Read source lines per file (cache)
        file_lines: dict[str, list[str]] = {}
        for node in nodes:
            if node.file_path not in file_lines:
                fp = root / node.file_path
                if fp.exists():
                    file_lines[node.file_path] = fp.read_text(encoding="utf-8", errors="replace").splitlines()

        chunks = chunk_nodes(nodes, file_lines)
        if not chunks:
            return

        batch_size = cfg.embedder.batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.header + "\n" + c.body for c in batch]
            vectors = embedder.embed_documents(texts)
            vstore.upsert(collection, list(zip(batch, vectors)))
            log.info("Embedded %d/%d chunks", min(i + batch_size, len(chunks)), len(chunks))

    def _resolve_references(self, parsed: ParsedFile, root: Path):
        from cce.graph.schema import Language  # noqa: PLC0415

        if parsed.language == Language.PYTHON:
            from cce.parsers.python_resolver import resolve_python_file  # noqa: PLC0415
            return resolve_python_file(parsed, root)
        else:
            from cce.parsers.js_resolver import resolve_js_file  # noqa: PLC0415
            return resolve_js_file(parsed, root)

    # ------------------------------------------------------------------
    # Convenience helpers (used by stores/CLI)
    # ------------------------------------------------------------------

    @property
    def symbol_store(self) -> SymbolStore:
        self._init_stores()
        return self._sym  # type: ignore[return-value]

    @property
    def lexical_store(self) -> LexicalStore:
        self._init_stores()
        return self._lex  # type: ignore[return-value]

    @property
    def graph_store(self) -> SQLiteGraphStore:
        self._init_stores()
        return self._graph  # type: ignore[return-value]
