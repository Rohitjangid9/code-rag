"""Phases 1-7 — IndexPipeline: walk → hash → lex → parse → framework-extract → resolve → graph → embed.

F28: per-file parsing is executed in a ``ThreadPoolExecutor`` (configurable via
     ``CCE_INDEXER__WORKERS``).  Pure-computation work (AST parse, framework
     extract, Jedi resolve) runs in parallel; all store writes are serialised
     on the main thread to avoid SQLite contention.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cce.config import Settings, get_settings
from cce.extractors.cli_extractor import CLIExtractor
from cce.extractors.django_extractor import DjangoExtractor
from cce.extractors.fastapi_extractor import FastAPIExtractor
from cce.extractors.framework_detector import detect_frameworks, file_belongs_to
from cce.extractors.langgraph_extractor import LangGraphExtractor
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


@dataclass
class _FileParseResult:
    """Holds everything computed during the pure-parse phase (F28).

    All fields are plain Python data (no store references) so the result can
    be passed safely from a worker thread to the main thread for serial writes.
    """

    wf: Any                    # WalkedFile
    rel: str
    source: str
    root: Any = None           # Path — passed through for Jedi resolution
    nodes: list = field(default_factory=list)         # list[Node]
    raw_edges: list = field(default_factory=list)     # list[RawEdge]
    router_prefixes: dict = field(default_factory=dict)
    sym_bodies: list = field(default_factory=list)    # [(qname, ls, le, body)] for F26
    error: str | None = None


_FRAMEWORK_EXTRACTORS = [
    DjangoExtractor(), FastAPIExtractor(), ReactExtractor(),
    CLIExtractor(), LangGraphExtractor(),
]


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
        self._router_prefixes: dict[str, str] = {}

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

    def run(
        self,
        root: Path,
        layers: list[str] | None = None,
        include_dirs: set[str] | None = None,
        skip_dirs: set[str] | None = None,
    ) -> IndexStats:
        """Run the full pipeline over *root*. *layers* filters which phases run.

        F-M6: ``include_dirs`` un-skips entries from the walker's soft-skip
        set (e.g. ``{"migrations"}`` to index Django migrations); ``skip_dirs``
        adds further directory names to ignore.
        """
        layers = layers or ["lexical", "symbols", "graph", "framework"]
        t0 = time.monotonic()
        self._init_stores()

        # F-M15: scope every ``_node_id_from_qname`` call made during this
        # pipeline run to the current repo so symbol IDs don't collide with
        # other repos sharing the same DB (centralised mode).  The salt is the
        # resolved repo root path — deterministic and unique per codebase.
        from cce.parsers.tree_sitter_parser import (  # noqa: PLC0415
            reset_repo_salt, set_repo_salt,
        )
        salt = str(Path(root).resolve())
        salt_token = set_repo_salt(salt)

        try:
            return self._run_inner(root, layers, include_dirs, skip_dirs, stats=IndexStats(root=str(root)), t0=t0)
        finally:
            reset_repo_salt(salt_token)

    def _run_inner(
        self,
        root: Path,
        layers: list[str],
        include_dirs: set[str] | None,
        skip_dirs: set[str] | None,
        stats: IndexStats,
        t0: float,
    ) -> IndexStats:
        """Body of :meth:`run` — factored out so the repo-salt context manager
        in :meth:`run` wraps every phase in a single ``try/finally``.
        """
        # Phase 1 — walk + change detection
        log.info("Walking %s …", root)
        walked: list[WalkedFile] = list(
            walk_repo(root, skip_dirs=skip_dirs, include_dirs=include_dirs)
        )
        stats.files_total = len(walked)

        changes: ChangeSet = detect_changes(walked, self._db, root)  # type: ignore[arg-type]
        stats.files_new = len(changes.new)
        stats.files_changed = len(changes.changed)
        stats.files_deleted = len(changes.deleted)

        for rel in changes.deleted:
            delete_file_records(rel, self._db)  # type: ignore[arg-type]

        # Phase 6 — detect frameworks once for the repo.  F-M10: reuse the
        # already-walked Python files so detection never descends into
        # ``node_modules`` / ``.venv`` and uses AST-level import checks.
        active_frameworks: set[FrameworkTag] = set()
        if "framework" in layers:
            py_files = [wf.path for wf in walked if wf.path.suffix == ".py"]
            active_frameworks = detect_frameworks(root, python_files=py_files)
            if active_frameworks:
                log.info("Detected frameworks: %s", {f.value for f in active_frameworks})

        # F28: parse files in parallel, apply writes serially
        workers = self._settings.indexer.workers or None  # None = os.cpu_count()
        self._index_files_parallel(
            changes.to_index, root, layers, stats, active_frameworks, workers
        )

        # F4: stamp effective_path on Route nodes using cross-file router prefixes
        if self._router_prefixes:
            self._stamp_effective_paths()

        # F-M13: cross-stack linker — resolve pending api_refs into CALLS_API
        # edges once every backend route is in the symbols table.
        if "graph" in layers or "framework" in layers:
            from cce.extractors.api_linker import link_api_references  # noqa: PLC0415
            created = link_api_references(self._db.conn)  # type: ignore[union-attr]
            stats.edges_indexed += created

        # Phase 7 — semantic embedding
        if "semantic" in layers:
            self._index_semantic(root, stats)

        stats.elapsed_s = time.monotonic() - t0
        log.info(
            "Done: %d new, %d changed, %d deleted, %d symbols, %d edges in %.1fs",
            stats.files_new, stats.files_changed, stats.files_deleted,
            stats.symbols_indexed, stats.edges_indexed, stats.elapsed_s,
        )
        # F12: write index manifest for startup health checks and cce doctor
        # F-M8: include frameworks and language breakdown so the agent prompt
        # can surface them without re-walking the repo at query time.
        languages = sorted({wf.language.value for wf in walked})
        framework_values = sorted({f.value for f in active_frameworks})
        self._write_manifest(root, stats, layers, languages, framework_values)
        return stats

    def _write_manifest(
        self,
        root: Path,
        stats: IndexStats,
        layers: list[str],
        languages: list[str] | None = None,
        frameworks: list[str] | None = None,
    ) -> None:
        """Write .cce/index.json with key metadata for startup health checks."""
        import json as _json  # noqa: PLC0415
        import datetime  # noqa: PLC0415
        try:
            commit_sha: str | None = None
            try:
                import subprocess as _sp  # noqa: PLC0415
                r = _sp.run(
                    ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    commit_sha = r.stdout.strip()
            except Exception:  # noqa: BLE001
                pass

            manifest = {
                "root": str(root),
                "indexed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "commit_sha": commit_sha,
                "file_count": stats.files_total,
                "symbol_count": stats.symbols_indexed,
                "edge_count": stats.edges_indexed,
                "languages": languages or [],
                "frameworks": frameworks or [],
                "layers": layers,
                "schema_version": self._settings.schema_version,
                "db_path": str(self._settings.paths.sqlite_db),
                "qdrant_path": str(self._settings.paths.qdrant_path),
            }
            manifest_path = self._settings.paths.data_dir / "index.json"
            manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
            log.debug("Wrote index manifest to %s", manifest_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to write index manifest: %s", exc)

    # ── F28: parallel indexing ────────────────────────────────────────────────

    def _index_files_parallel(
        self,
        files: list,
        root: Path,
        layers: list[str],
        stats: IndexStats,
        active_frameworks,
        workers: int | None,
    ) -> None:
        """Parse files in a thread pool; apply writes in two serial passes (F28).

        Pass 1 – collect all parse results and commit symbols/lexical data so
                  every symbol is visible in the DB before edge resolution runs.
        Pass 2 – run Jedi reference resolution now that the full symbol table is
                  available, preventing race-condition drops where graph.py
                  references nodes.py symbols that weren't indexed yet.
        """
        if not files:
            return

        results: list[_FileParseResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._parse_file_worker, wf, root, layers, active_frameworks
                ): wf
                for wf in files
            }
            for future in as_completed(futures):
                wf = futures[future]
                try:
                    result = future.result()
                    # Pass 1: symbols + lexical only (no Jedi edge resolution yet)
                    self._apply_symbols(result, layers, stats)
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    msg = f"error indexing {wf.rel_path}: {exc}"
                    log.warning(msg)
                    stats.errors.append(msg)

        # Pass 2: all symbols committed — now safe to resolve cross-file references
        for result in results:
            try:
                self._apply_edges(result, layers, stats)
            except Exception as exc:  # noqa: BLE001
                msg = f"error resolving edges for {result.rel}: {exc}"
                log.warning(msg)
                stats.errors.append(msg)

    def _parse_file_worker(
        self,
        wf: WalkedFile,
        root: Path,
        layers: list[str],
        active_frameworks,
    ) -> _FileParseResult:
        """Pure-computation phase: tree-sitter AST parse + framework extraction (F28).

        Only thread-safe operations run here (tree-sitter, regex, framework
        extractors).  Jedi reference resolution and all store writes are done
        serially in ``_apply_parsed`` to avoid race conditions.
        """
        source = wf.path.read_text(encoding="utf-8", errors="replace")
        # F-WIN: always use POSIX (forward-slash) paths so the "/"
        # presence check and split("/") in parsers / extractors work on
        # Windows as well as Linux/macOS.
        rel = wf.rel_path.as_posix()
        result = _FileParseResult(wf=wf, rel=rel, source=source, root=root)  # rel is POSIX

        parsed: ParsedFile | None = None
        if "symbols" in layers or "graph" in layers or "framework" in layers:
            parsed = self._parser.parse(wf.path, rel, wf.language, source)
            result.nodes.extend(parsed.nodes)
            # F26: collect per-symbol bodies for lexical store
            if "lexical" in layers:
                src_lines = source.splitlines()
                for node in parsed.nodes:
                    if node.line_start and node.line_end and node.line_end > node.line_start:
                        body = "\n".join(src_lines[node.line_start - 1: node.line_end])
                        result.sym_bodies.append(
                            (node.qualified_name, node.line_start, node.line_end, body)
                        )

        if "framework" in layers and parsed is not None and active_frameworks:
            for extractor in _FRAMEWORK_EXTRACTORS:
                if extractor.can_handle(wf.path, source):
                    fw_data = extractor.extract(wf.path, rel, source)
                    result.nodes.extend(fw_data.nodes)
                    result.router_prefixes.update(fw_data.router_prefixes)
                    parsed.raw_edges.extend(fw_data.raw_edges)

        if "graph" in layers and parsed is not None:
            # Store raw_edges from AST (no Jedi — that runs serially in _apply_parsed)
            result.raw_edges.extend(parsed.raw_edges)
            result._parsed = parsed  # stash ParsedFile for _apply_parsed  # noqa: SLF001

        return result

    def _apply_symbols(self, result: _FileParseResult, layers: list[str], stats: IndexStats) -> None:
        """Pass 1: write lexical data + symbols only.  No edge resolution yet.

        Called for every file in the thread-pool completion loop so all symbols
        are committed to the DB before cross-file Jedi resolution runs in Pass 2.
        """
        rel = result.rel
        source = result.source
        cfg = self._settings.indexer

        # Phase 2 — lexical (file-level)
        if "lexical" in layers:
            self._lex.upsert(rel, source)  # type: ignore[union-attr]
            for qname, ls, le, body in result.sym_bodies:
                self._lex.upsert_symbol(rel, qname, ls, le, body)  # type: ignore[union-attr]

        # Phase 3 — symbol store
        sym_count = len(result.nodes)
        if result.nodes:
            self._sym.delete_for_file(rel)  # type: ignore[union-attr]
            self._sym.upsert_many(result.nodes)  # type: ignore[union-attr]
            stats.symbols_indexed += sym_count

        # Router prefixes (framework extractor)
        self._router_prefixes.update(result.router_prefixes)

        if cfg.verbose:
            log.debug(
                "[pass1] %-55s  symbols=%d  ast_edges=%d",
                rel, sym_count, len(result.raw_edges),
            )

    def _apply_edges(self, result: _FileParseResult, layers: list[str], stats: IndexStats) -> None:
        """Pass 2: Jedi resolution + graph edges.

        Runs AFTER all symbols from all files are committed so cross-file
        ``resolve_qname`` calls succeed regardless of processing order.
        """
        rel = result.rel
        parsed: ParsedFile | None = getattr(result, "_parsed", None)
        cfg = self._settings.indexer

        # Phase 4/5 — Jedi resolution + graph edges (serial — Jedi is not thread-safe)
        if "graph" in layers and parsed is not None:
            raw_edges = list(result.raw_edges)
            jedi_edges = self._resolve_references(parsed, root=result.root)
            raw_edges += jedi_edges

            log.debug(
                "[pass2] %-55s  ast=%d  jedi=%d  total=%d",
                rel, len(result.raw_edges), len(jedi_edges), len(raw_edges),
            )

            if raw_edges:
                self._graph.delete_for_file(rel)  # type: ignore[union-attr]
                # F-M13: purge any previous API refs recorded for this file
                self._db.conn.execute(  # type: ignore[union-attr]
                    "DELETE FROM api_refs WHERE file_path = ?", (rel,),
                )

                # Counters for edge-kind breakdown (edge_debug)
                written: dict[str, int] = {}
                dropped = 0
                api_deferred = 0

                for re_ in raw_edges:
                    # F-M13: defer api:// REFERENCES to the linker post-pass
                    if re_.dst_qualified_name.startswith("api:"):
                        self._db.conn.execute(  # type: ignore[union-attr]
                            "INSERT INTO api_refs "
                            "(src_id, path, method, file_path, line, confidence) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                re_.src_id,
                                re_.dst_qualified_name.removeprefix("api:"),
                                None,
                                re_.file_path,
                                re_.line,
                                re_.confidence,
                            ),
                        )
                        api_deferred += 1
                        continue

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
                        kind_key = re_.kind.value
                        written[kind_key] = written.get(kind_key, 0) + 1
                        # F-LG-META: merge dst_meta_patch into the resolved symbol
                        if re_.dst_meta_patch:
                            self._sym.merge_meta(dst_id, re_.dst_meta_patch)  # type: ignore[union-attr]
                    else:
                        dropped += 1
                        log.debug(
                            "[pass2] unresolvable dst=%s  kind=%s  src=%s",
                            re_.dst_qualified_name, re_.kind.value, rel,
                        )

                # Per-file summary when verbose is on
                if cfg.verbose:
                    total_written = sum(written.values())
                    log.debug(
                        "[pass2] %-55s  written=%d  dropped=%d  api_deferred=%d",
                        rel, total_written, dropped, api_deferred,
                    )

                # Detailed edge-kind breakdown when edge_debug is on
                if cfg.edge_debug and written:
                    breakdown = "  ".join(f"{k}={v}" for k, v in sorted(written.items()))
                    log.debug("[pass2:kinds] %s  →  %s", rel, breakdown)

    # ── Phase 7 — semantic embedding ──────────────────────────────────────────

    def _index_semantic(self, root: Path, stats: IndexStats) -> None:
        """Chunk all symbols, embed with the configured embedder, upsert to Qdrant."""
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

        # F22: delete stale vectors for files in the change set before upserting
        changed_node_ids = [c.node_id for c in chunks]
        if changed_node_ids:
            vstore.delete_for_node_ids(collection, changed_node_ids)
            log.debug("Deleted stale vectors for %d nodes", len(changed_node_ids))

        # F21: fetch existing content hashes; skip chunks whose content is unchanged
        existing_hashes = vstore.get_existing_hashes(collection, changed_node_ids)
        chunks_to_embed = [
            c for c in chunks
            if existing_hashes.get(c.node_id) != c.content_hash
        ]
        skipped = len(chunks) - len(chunks_to_embed)
        if skipped:
            log.info("Skipped %d unchanged chunks (F21 hash-dedup)", skipped)

        batch_size = cfg.embedder.batch_size
        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i: i + batch_size]
            texts = [c.header + "\n" + c.body for c in batch]
            vectors = embedder.embed_documents(texts)
            vstore.upsert(collection, list(zip(batch, vectors)))
            log.info("Embedded %d/%d chunks", min(i + batch_size, len(chunks_to_embed)), len(chunks_to_embed))

    def _stamp_effective_paths(self) -> None:
        """After framework extraction, prepend cross-file include_router prefixes."""
        import json as _json  # noqa: PLC0415
        from cce.extractors.fastapi_extractor import _join_paths  # noqa: PLC0415

        conn = self._db.conn  # type: ignore[union-attr]
        rows = conn.execute(
            "SELECT id, meta FROM symbols WHERE kind = 'Route'"
        ).fetchall()

        for row in rows:
            meta = _json.loads(row["meta"] or "{}")
            router_var = meta.get("router_var", "")
            router_module = meta.get("router_module", "")
            path = meta.get("path", "")
            effective_path = path
            # Apply only cross-file prefixes (same-file prefixes are already baked into path).
            for prefix_key, prefix in self._router_prefixes.items():
                mod, var = prefix_key.rsplit(".", 1)
                if var == router_var and mod != router_module:
                    effective_path = _join_paths(prefix, effective_path)
                    break
            if effective_path != path:
                meta["effective_path"] = effective_path
                conn.execute(
                    "UPDATE symbols SET meta = ? WHERE id = ?",
                    (_json.dumps(meta), row["id"]),
                )
        conn.commit()

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
