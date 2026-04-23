"""Typer CLI entrypoint — all commands through Phase 5 are implemented."""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which chokes on rich glyphs like ✓ / ⚠.
# Switch stdout/stderr to UTF-8 before any rich import writes to them.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import os

import typer
from rich.console import Console
from rich.table import Table

from cce.config import get_settings
from cce.logging import get_logger

app = typer.Typer(
    name="cce",
    help="Code Context Engine — index and query codebases (Django / FastAPI / React).",
    no_args_is_help=True,
)
console = Console()
log = get_logger(__name__)


# ── F-M4: global --repo option (auto-detected when unset) ─────────────────────

@app.callback()
def _global_opts(
    repo: Path | None = typer.Option(
        None,
        "--repo",
        "-r",
        help="Path to the indexed repo. Auto-detected by walking up for .cce/ if unset.",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Set CCE_REPO_ROOT so every subcommand binds Settings to the right repo."""
    if repo is not None:
        os.environ["CCE_REPO_ROOT"] = str(repo.resolve())


# ── Phase 1 ────────────────────────────────────────────────────────────────────

@app.command()
def scan(path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    """Walk a codebase and print a file inventory grouped by language."""
    from collections import defaultdict  # noqa: PLC0415

    from cce.walker import walk_repo  # noqa: PLC0415

    groups: dict[str, list[str]] = defaultdict(list)
    for wf in walk_repo(path):
        groups[wf.language.value].append(str(wf.rel_path))

    tbl = Table(title=f"Files in {path}", show_lines=False)
    tbl.add_column("Language", style="cyan")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Example", style="dim")
    for lang, fpaths in sorted(groups.items()):
        tbl.add_row(lang, str(len(fpaths)), fpaths[0] if fpaths else "")
    console.print(tbl)


# ── Phase 1-5 (full pipeline) ──────────────────────────────────────────────────

@app.command()
def index(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    layers: str = typer.Option(
        "lexical,symbols,graph,framework",
        help="Comma-separated: lexical, symbols, graph, framework, semantic",
    ),
    include: str = typer.Option(
        "",
        "--include",
        help="Comma-separated soft-skip dirs to include (e.g. 'migrations,vendor').",
    ),
    skip: str = typer.Option(
        "",
        "--skip",
        help="Comma-separated extra dir names to exclude (beyond built-in skips).",
    ),
) -> None:
    """Build index layers for the given codebase root.

    Add 'semantic' to also embed with nomic-embed-code into Qdrant (requires ~7GB VRAM).
    """
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    # F-M4: anchor the pipeline Settings to the repo being indexed so .cce/
    # is created inside *path*, not the CWD.
    settings = get_settings(repo_root=path)
    layer_list = [la.strip() for la in layers.split(",")]
    include_set = {s.strip() for s in include.split(",") if s.strip()}
    skip_set = {s.strip() for s in skip.split(",") if s.strip()}
    with console.status(f"Indexing {path} …"):
        stats = IndexPipeline(settings=settings).run(
            path, layer_list, include_dirs=include_set or None,
            skip_dirs=skip_set or None,
        )
    console.print(f"[green]✓[/] {stats.files_new} new, {stats.files_changed} changed, "
                  f"{stats.files_deleted} deleted | [bold]{stats.symbols_indexed}[/] symbols "
                  f"| [bold]{stats.edges_indexed}[/] edges | {stats.elapsed_s:.1f}s")
    if stats.errors:
        for e in stats.errors[:5]:
            console.print(f"  [red]![/] {e}")


# ── Phase 2 ────────────────────────────────────────────────────────────────────

@app.command()
def search(
    q: str = typer.Argument(...),
    mode: str = typer.Option("lexical", help="lexical | symbols | hybrid"),
    k: int = typer.Option(10),
) -> None:
    """Search the indexed codebase."""
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    pipeline = IndexPipeline()
    if mode in ("lexical", "hybrid"):
        hits = pipeline.lexical_store.search(q, k=k)
        tbl = Table(title=f"Lexical: {q!r}", show_lines=False)
        tbl.add_column("Path", style="cyan")
        tbl.add_column("Snippet")
        for h in hits:
            tbl.add_row(h.path, h.snippet[:120])
        console.print(tbl)
    if mode in ("symbols", "hybrid"):
        sym_hits = pipeline.symbol_store.search(q, k=k)
        tbl = Table(title=f"Symbols: {q!r}", show_lines=False)
        tbl.add_column("QName", style="cyan")
        tbl.add_column("Kind")
        tbl.add_column("File")
        for h in sym_hits:
            tbl.add_row(h.node.qualified_name, h.node.kind.value, h.node.file_path)
        console.print(tbl)


# ── Phase 3 ────────────────────────────────────────────────────────────────────

@app.command()
def symbols(
    path: Path = typer.Argument(..., exists=True),
    kind: str = typer.Option("", help="Filter by kind (Class, Function, Method, Component …)"),
) -> None:
    """List all symbols in a file or directory.

    Paths stored in the index are relative to the root passed to ``cce index``,
    so this command matches on any suffix of the stored ``file_path``.
    """
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    sym_store = IndexPipeline().symbol_store
    conn = sym_store._db.conn

    # Build a list of candidate rel-path strings to match against file_path suffix.
    p = path.resolve()
    if p.is_file():
        # Try the bare name, the full OS-style path, and the posix-style path.
        candidates = {p.name, str(path), p.as_posix(), str(path).replace("\\", "/")}
        where = " OR ".join(["file_path = ?"] * len(candidates) + ["file_path LIKE ?"] * len(candidates))
        params = list(candidates) + [f"%/{c}" for c in candidates]
    else:
        # Directory: match any file whose path starts with or ends under this dir.
        dir_str = str(path).replace("\\", "/").rstrip("/")
        where = "file_path LIKE ? OR file_path LIKE ?"
        params = [f"{dir_str}/%", f"%/{Path(dir_str).name}/%"]

    rows = conn.execute(
        f"SELECT * FROM symbols WHERE {where} ORDER BY file_path, line_start",
        params,
    ).fetchall()

    from cce.index.symbol_store import _row_to_node  # noqa: PLC0415
    nodes = [_row_to_node(r) for r in rows]
    if kind:
        nodes = [n for n in nodes if n.kind.value.lower() == kind.lower()]
    if not nodes and p.is_dir():
        # Fallback: list every indexed symbol (user asked about a repo dir that
        # was indexed with a different relative root).
        rows = conn.execute("SELECT * FROM symbols ORDER BY file_path, line_start").fetchall()
        nodes = [_row_to_node(r) for r in rows]
        if kind:
            nodes = [n for n in nodes if n.kind.value.lower() == kind.lower()]

    tbl = Table(title=f"Symbols in {path}  ({len(nodes)} total)", show_lines=False)
    tbl.add_column("Kind", style="cyan", width=14)
    tbl.add_column("Qname", style="bold")
    tbl.add_column("File")
    tbl.add_column("Line", justify="right")
    tbl.add_column("Signature")
    for n in nodes:
        tbl.add_row(
            n.kind.value, n.qualified_name, n.file_path,
            str(n.line_start), (n.signature or "")[:60],
        )
    console.print(tbl)


@app.command("get")
def get_symbol(qname: str = typer.Argument(...)) -> None:
    """Show full details for a qualified-name symbol."""
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    node = IndexPipeline().symbol_store.get_by_qname(qname)
    if not node:
        console.print(f"[red]Not found:[/] {qname}")
        raise typer.Exit(1)
    console.print_json(node.model_dump_json(indent=2))


# ── Phase 4 ────────────────────────────────────────────────────────────────────

@app.command()
def callers(qname: str = typer.Argument(...)) -> None:
    """List all symbols that call the given qualified name."""
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    pipeline = IndexPipeline()
    node = pipeline.symbol_store.get_by_qname(qname)
    if not node:
        console.print(f"[red]Not found:[/] {qname}")
        raise typer.Exit(1)
    results = pipeline.graph_store.find_callers(node.id)
    for n in results:
        console.print(f"  [cyan]{n.qualified_name}[/] ({n.file_path}:{n.line_start})")
    if not results:
        console.print("[dim]No callers found.[/]")


@app.command()
def refs(qname: str = typer.Argument(...)) -> None:
    """List all reference edges pointing to the given qualified name."""
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    pipeline = IndexPipeline()
    node = pipeline.symbol_store.get_by_qname(qname)
    if not node:
        console.print(f"[red]Not found:[/] {qname}")
        raise typer.Exit(1)
    edges = pipeline.graph_store.find_references(node.id)
    for e in edges:
        console.print(f"  {e.kind.value}  {e.src_id} → {e.dst_id} ({e.location})")
    if not edges:
        console.print("[dim]No references found.[/]")


# ── Phase 5 ────────────────────────────────────────────────────────────────────

@app.command()
def neighborhood(
    qname: str = typer.Argument(...),
    depth: int = typer.Option(2),
) -> None:
    """Print the N-hop graph neighborhood of the given symbol."""
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    pipeline = IndexPipeline()
    node = pipeline.symbol_store.get_by_qname(qname)
    if not node:
        console.print(f"[red]Not found:[/] {qname}")
        raise typer.Exit(1)
    subgraph = pipeline.graph_store.get_neighborhood(node.id, depth=depth)
    console.print(f"[bold]Neighborhood of {qname}[/] (depth={depth}): "
                  f"{len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
    for n in subgraph.nodes:
        console.print(f"  [cyan]{n.kind.value}[/] {n.qualified_name}")
    for e in subgraph.edges:
        console.print(f"  [dim]{e.src_id} --{e.kind.value}--> {e.dst_id}[/]")


# ── Phase 8 ────────────────────────────────────────────────────────────────────

@app.command()
def query(
    q: str = typer.Argument(..., help="Natural-language or identifier query."),
    mode: str = typer.Option("hybrid", help="lexical | hybrid | semantic"),
    k: int = typer.Option(10),
) -> None:
    """Run the hybrid retriever (Phase 8)."""
    from cce.retrieval.tools import search_code  # noqa: PLC0415
    from typing import Literal  # noqa: PLC0415
    hits = search_code(q, mode=mode, k=k)  # type: ignore[arg-type]
    tbl = Table(title=f"Results for {q!r} [{mode}]", show_lines=False)
    tbl.add_column("Score", justify="right", width=7)
    tbl.add_column("Prov", width=8)
    tbl.add_column("Symbol / Path", style="cyan")
    tbl.add_column("Line", justify="right", width=5)
    tbl.add_column("Snippet")
    for h in hits:
        sym = h.node.qualified_name if h.node else h.path
        tbl.add_row(f"{h.score:.3f}", h.provenance, sym, str(h.line_start), h.snippet[:60])
    console.print(tbl)


# ── Phase 10 ───────────────────────────────────────────────────────────────────

@app.command()
def watch(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    layers: str = typer.Option("lexical,symbols,graph,framework"),
    debounce: float = typer.Option(1.0, help="Debounce seconds between file saves."),
) -> None:
    """Watch a codebase and re-index changed files incrementally (Phase 10)."""
    import signal  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415
    from cce.watcher.file_watcher import FileWatcher  # noqa: PLC0415

    pipeline = IndexPipeline()
    console.print(f"[green]Watching[/] {path} (layers: {layers})")
    console.print("Press [bold]Ctrl-C[/] to stop.")

    with FileWatcher(pipeline, path, debounce_s=debounce):
        try:
            signal.pause()  # unix; on Windows the except below handles it
        except (AttributeError, KeyboardInterrupt):
            pass


# ── Phase 11 ───────────────────────────────────────────────────────────────────

@app.command("export-scip")
def export_scip(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    out: Path = typer.Option(Path("index.scip.json"), help="Output JSON file."),
) -> None:
    """Export the symbol index in SCIP JSON format (Phase 11)."""
    from cce.scip.emitter import SCIPEmitter  # noqa: PLC0415
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    pipeline = IndexPipeline()
    emitter = SCIPEmitter(pipeline.symbol_store, pipeline.graph_store)
    index = emitter.emit(root=path)
    out.write_text(index.to_json(), encoding="utf-8")
    console.print(f"[green]✓[/] SCIP index written to [bold]{out}[/] "
                  f"({len(index.documents)} documents, "
                  f"{sum(len(d.symbols) for d in index.documents)} symbols)")


# ── Phase 12 ───────────────────────────────────────────────────────────────────

@app.command()
def eval(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    queries: Path = typer.Option(..., exists=True, help="YAML eval query file."),
    k: int = typer.Option(10, help="Recall/MRR cutoff."),
) -> None:
    """Run the retrieval eval harness (Phase 12)."""
    from cce.eval.harness import EvalHarness  # noqa: PLC0415
    from cce.eval.dataset import EvalDataset  # noqa: PLC0415

    dataset = EvalDataset.from_yaml(queries)
    harness = EvalHarness(root=path, k=k)
    report = harness.run(dataset)
    console.print(report.rich_table())


@app.command(name="eval-agent")
def eval_agent(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    dataset: Path = typer.Option(
        None, "--dataset", "-d", exists=False,
        help="YAML eval query file.  Defaults to cce/eval/datasets/core.yaml.",
    ),
    k: int = typer.Option(10, "--k", help="Recall/MRR cutoff."),
    scorecard: Path = typer.Option(
        None, "--scorecard", "-s",
        help="Output path for scorecard.json (F14).  Defaults to .cce/scorecard.json.",
    ),
    mrr_threshold: float = typer.Option(0.50, "--mrr", help="MRR CI gate threshold."),
    recall_threshold: float = typer.Option(0.50, "--recall", help="Recall CI gate threshold."),
    ndcg_threshold: float = typer.Option(0.50, "--ndcg", help="nDCG CI gate threshold."),
    fail_on_regression: bool = typer.Option(False, "--fail", "-f", help="Exit 1 if thresholds not met."),
) -> None:
    """Run the retrieval eval harness and write a scorecard.json (F14).

    Used as a CI gate — ``--fail`` causes exit code 1 if any threshold is missed.

    \b
    Examples::

        cce eval-agent /path/to/repo
        cce eval-agent /path/to/repo --dataset my_queries.yaml --fail
        cce eval-agent /path/to/repo --mrr 0.7 --recall 0.65 --fail
    """
    from cce.eval.harness import EvalHarness as _EvalHarness  # noqa: PLC0415
    from cce.eval.dataset import EvalDataset as _EvalDataset  # noqa: PLC0415

    # Resolve dataset path
    if dataset is None:
        dataset = Path(__file__).parent / "eval" / "datasets" / "core.yaml"
    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/]")
        raise typer.Exit(1)

    # Resolve scorecard path
    cfg = get_settings()
    scorecard_path = scorecard or (cfg.paths.data_dir / "scorecard.json")

    ds = _EvalDataset.from_yaml(dataset)
    thresholds = {"mrr": mrr_threshold, "recall": recall_threshold, "ndcg": ndcg_threshold}
    harness = _EvalHarness(root=path, k=k, thresholds=thresholds)
    report = harness.run(ds)
    console.print(report.rich_table())

    passed = report.write_scorecard(
        scorecard_path, dataset_name=dataset.stem, thresholds=thresholds
    )
    console.print(f"[dim]Scorecard → {scorecard_path}[/]")

    if not passed:
        console.print("[yellow]⚠  One or more CI thresholds not met — see scorecard.[/]")
        if fail_on_regression:
            raise typer.Exit(1)


def _check_tool_call_support(provider: str, model: str) -> tuple[bool, str]:
    """Return (is_compatible, human_detail) for the configured agent LLM.

    Rough allow-list based on published provider docs. Conservative by design
    — we warn (not fail) when unsure so a user with a fine-tuned model can
    override. Any model that cannot emit structured tool_calls will trigger
    the planner's leak-recovery path at runtime (see cce.agents.nodes).
    """
    if provider == "openai":
        ok_markers = (
            "gpt-4", "gpt-5", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
            "o1", "o3", "o4",
        )
        if any(mk in model for mk in ok_markers) or model == "":
            return True, "OpenAI function-calling supported"
        return False, f"model {model!r} may not support tools — use gpt-4* / gpt-5* / gpt-3.5-turbo-1106+"
    if provider == "anthropic":
        if "claude-3" in model or "claude-4" in model or model == "":
            return True, "Anthropic tool_use supported"
        return False, f"model {model!r} — use claude-3* or newer"
    if provider == "ollama":
        tool_capable = ("llama3.1", "llama3.2", "mistral-nemo", "firefunction",
                        "command-r", "qwen2.5", "hermes-3")
        if any(mk in model for mk in tool_capable):
            return True, "Ollama model listed as tool-capable"
        return False, (f"model {model!r} may emit tool-call text instead of structured "
                       "tool_calls — planner will try to recover, but prefer llama3.1 / qwen2.5 / hermes-3")
    return False, f"unknown provider {provider!r}"


@app.command()
def doctor() -> None:
    """Check all critical dependencies and configuration."""
    import importlib  # noqa: PLC0415
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    tbl = Table(title="cce doctor", show_lines=True)
    tbl.add_column("Check", style="bold", width=34)
    tbl.add_column("Status", width=9)
    tbl.add_column("Detail")

    def _ok(name: str, detail: str = "") -> None:
        tbl.add_row(name, "[green]✓ PASS[/]", detail)
    def _warn(name: str, detail: str = "") -> None:
        tbl.add_row(name, "[yellow]⚠ WARN[/]", detail)
    def _fail(name: str, detail: str = "") -> None:
        tbl.add_row(name, "[red]✗ FAIL[/]", detail)

    # tree-sitter core + individual language bindings (tree-sitter >= 0.23 API)
    try:
        from tree_sitter import Language, Parser  # noqa: PLC0415
        _ok("tree-sitter core", f"{Language.__module__}")
    except Exception as e:
        _fail("tree-sitter core", str(e)[:80])

    _LANG_CHECKS = [
        ("tree-sitter python", "tree_sitter_python", "language"),
        ("tree-sitter javascript", "tree_sitter_javascript", "language"),
        ("tree-sitter typescript", "tree_sitter_typescript", "language_typescript"),
        ("tree-sitter tsx", "tree_sitter_typescript", "language_tsx"),
    ]
    for label, mod_name, attr in _LANG_CHECKS:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            getattr(mod, attr)
            _ok(label)
        except Exception as e:
            _fail(label, str(e)[:80])

    # SQLite
    try:
        from cce.config import get_settings as _gs  # noqa: PLC0415
        from cce.index.db import get_db as _gdb  # noqa: PLC0415
        cfg = _gs(); cfg.paths.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
        _gdb(cfg.paths.sqlite_db).conn.execute("SELECT 1")
        _ok("SQLite DB", str(cfg.paths.sqlite_db))
    except Exception as e:
        _fail("SQLite DB", str(e)[:80])

    # Qdrant
    try:
        from qdrant_client import QdrantClient as _QC  # noqa: PLC0415
        from cce.config import get_settings as _gs  # noqa: PLC0415
        cfg = _gs()
        if cfg.qdrant.mode == "embedded":
            cfg.paths.qdrant_path.mkdir(parents=True, exist_ok=True)
            _QC(path=str(cfg.paths.qdrant_path))
            _ok("Qdrant (embedded)", str(cfg.paths.qdrant_path))
        else:
            client = _QC(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key)
            client.get_collections()
            _ok("Qdrant (remote)", cfg.qdrant.url or "")
    except Exception as e:
        _fail("Qdrant", str(e)[:80])

    # Re-read .env file directly without caching side-effects
    def _env_val(var: str) -> str | None:
        # 1) real OS env  2) pydantic-settings .env  3) None
        if os.getenv(var):
            return os.getenv(var)
        try:
            from pathlib import Path  # noqa: PLC0415
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith(var + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
        return None

    # Embedder backend
    try:
        from cce.config import get_settings as _gs  # noqa: PLC0415
        from cce.embeddings.embedder import _resolve  # noqa: PLC0415
        cfg = _gs().embedder; model, dim = _resolve(cfg.backend, cfg.model_name, cfg.dim)
        if cfg.backend == "jina":
            importlib.import_module("sentence_transformers")
            _ok(f"Embedder ({cfg.backend})", f"{model}  dim={dim}  CPU-ready")
        elif cfg.backend == "openai":
            if _env_val("OPENAI_API_KEY"):
                _ok(f"Embedder ({cfg.backend})", f"{model}  dim={dim}")
            else:
                _warn(f"Embedder ({cfg.backend})", "OPENAI_API_KEY not set")
        elif cfg.backend == "nomic":
            try:
                import torch as _t  # noqa: PLC0415
                if _t.cuda.is_available():
                    gb = _t.cuda.get_device_properties(0).total_memory / 1e9
                    fn = _ok if gb >= 7 else _warn
                    fn(f"Embedder ({cfg.backend})", f"{model}  dim={dim}  GPU {gb:.1f}GB")
                else:
                    _warn(f"Embedder ({cfg.backend})", "No CUDA GPU — nomic on CPU is very slow")
            except ImportError:
                _fail(f"Embedder ({cfg.backend})", "torch not installed")
    except Exception as e:
        _fail("Embedder", str(e)[:80])

    # GPU
    try:
        import torch as _t  # noqa: PLC0415
        if _t.cuda.is_available():
            gb = _t.cuda.get_device_properties(0).total_memory / 1e9
            _ok("GPU (CUDA)", f"{_t.cuda.get_device_name(0)}  {gb:.1f} GB")
        elif _t.backends.mps.is_available():
            _ok("GPU (MPS)", "Apple Silicon")
        else:
            _warn("GPU", "No GPU — embedding runs on CPU")
    except ImportError:
        _warn("GPU check", "torch not installed")

    # API keys — also read from .env directly if not exported as real OS env var.
    for var, note in [
        ("OPENAI_API_KEY", "needed for openai embedder + agent"),
        ("ANTHROPIC_API_KEY", "needed if CCE_AGENT__LLM_PROVIDER=anthropic"),
    ]:
        val = _env_val(var)
        (_ok if val else _warn)(var, "set" if val else note)

    # Agent LLM tool-calling compatibility (P0-1 companion check).
    try:
        from cce.config import get_settings as _gs  # noqa: PLC0415
        ag = _gs().agent
        provider, model = ag.llm_provider, (ag.llm_model or "").lower()
        compatible, detail = _check_tool_call_support(provider, model)
        (_ok if compatible else _warn)(
            "Agent tool-calling",
            f"{provider}/{ag.llm_model or '(default)'} — {detail}",
        )
    except Exception as e:  # noqa: BLE001
        _warn("Agent tool-calling", str(e)[:80])

    # F12: Index manifest check
    try:
        from cce.config import get_settings as _gs  # noqa: PLC0415
        import json as _json  # noqa: PLC0415
        cfg = _gs()
        manifest_path = cfg.paths.data_dir / "index.json"
        if manifest_path.exists():
            m = _json.loads(manifest_path.read_text(encoding="utf-8"))
            mver = m.get("schema_version")
            detail = f"root={m.get('root','?')}  indexed_at={m.get('indexed_at','?')}"
            if mver != cfg.schema_version:
                _warn("Index manifest", f"schema_version mismatch (manifest={mver}, expected={cfg.schema_version})")
            else:
                _ok("Index manifest", detail)
        else:
            _warn("Index manifest", "not found — run `cce index <path>`")
    except Exception as e:
        _fail("Index manifest", str(e)[:80])

    # Python packages
    for pkg, note in [("watchdog","watcher"), ("ulid","IDs"), ("yaml","eval"),
                      ("langchain_core","LangChain"), ("langgraph","agent graph")]:
        try:
            importlib.import_module(pkg); _ok(f"pkg: {pkg}")
        except ImportError:
            _fail(f"pkg: {pkg}", note)

    # Node.js (future ts-morph)
    try:
        r = subprocess.run(["node","--version"], capture_output=True, text=True, timeout=5)
        (_ok if r.returncode == 0 else _warn)("Node.js", r.stdout.strip() + " (ts-morph sidecar, future)")
    except FileNotFoundError:
        _warn("Node.js", "not found — needed for future ts-morph TS resolver")

    console.print(tbl)


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
) -> None:
    """Start the FastAPI server.

    The MCP (Model Context Protocol) JSON-RPC endpoint is always mounted at
    POST /mcp and GET /mcp/tools — no flag required.
    """
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cce.server.app:create_app",
        host=host or settings.server.host,
        port=port or settings.server.port,
        factory=True,
        reload=False,
    )


# ── Phase 6 framework tools ───────────────────────────────────────────────────

@app.command("get-route")
def get_route_cmd(
    pattern_or_path: str = typer.Argument(..., help="URL pattern or concrete path."),
) -> None:
    """Resolve a URL pattern or concrete path to its handler + response model."""
    from cce.retrieval.tools import get_route  # noqa: PLC0415

    try:
        info = get_route(pattern_or_path)
    except KeyError as exc:
        console.print(f"[red]Not found:[/] {exc}")
        raise typer.Exit(1)
    console.print_json(info.model_dump_json(indent=2))


@app.command("get-api-flow")
def get_api_flow_cmd(
    route_or_component: str = typer.Argument(..., help="Route path, pattern, or component name."),
) -> None:
    """Return the UI → API → handler → model chain for the given anchor."""
    from cce.retrieval.tools import get_api_flow  # noqa: PLC0415

    flow = get_api_flow(route_or_component)
    console.print_json(flow.model_dump_json(indent=2))


@app.command()
def info() -> None:
    """Print resolved configuration."""
    settings = get_settings()
    console.print_json(settings.model_dump_json(indent=2))


@app.command("inspect-qdrant")
def inspect_qdrant(
    collection: str | None = typer.Option(None, help="Collection name (lists all if omitted)."),
    limit: int = typer.Option(5, help="Number of sample points to show per collection."),
) -> None:
    """Inspect Qdrant collections, counts, vector dims, and sample payloads."""
    from qdrant_client import QdrantClient  # noqa: PLC0415
    from cce.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    cfg = settings.qdrant

    if cfg.mode == "embedded":
        client = QdrantClient(path=str(settings.paths.qdrant_path))
    else:
        client = QdrantClient(url=cfg.url, api_key=cfg.api_key)

    try:
        collections = client.get_collections().collections
    except Exception as exc:
        console.print(f"[red]Could not connect to Qdrant:[/] {exc}")
        raise typer.Exit(1)

    if not collections:
        console.print("[yellow]No collections found in Qdrant.[/]")
        raise typer.Exit(0)

    # Summary table
    tbl = Table(title="Qdrant Collections", show_lines=False)
    tbl.add_column("Collection", style="cyan")
    tbl.add_column("Vector Size", justify="right")
    tbl.add_column("Points", justify="right")
    for c in collections:
        info = client.get_collection(c.name)
        count = client.count(c.name).count
        vec_cfg = info.config.params.vectors
        size = getattr(vec_cfg, "size", "N/A") if not isinstance(vec_cfg, dict) else "multi"
        tbl.add_row(c.name, str(size), str(count))
    console.print(tbl)

    target = collection or (collections[0].name if len(collections) == 1 else None)
    if not target:
        console.print("\n[dim]Use --collection NAME to inspect payloads.[/]")
        return

    console.print(f"\n[bold]Sample payloads from {target}:[/]")
    points, _ = client.scroll(target, limit=limit, with_vectors=False, with_payload=True)
    if not points:
        console.print("[dim]No points in collection.[/]")
        return

    for p in points:
        payload = p.payload or {}
        pt_tbl = Table(title=f"Point {p.id}", show_lines=False, width=100)
        pt_tbl.add_column("Field", style="cyan", width=18)
        pt_tbl.add_column("Value")
        for k, v in sorted(payload.items()):
            pt_tbl.add_row(k, str(v)[:200])
        console.print(pt_tbl)


@app.command()
def answer(
    query: str = typer.Argument(..., help="Natural-language question about the codebase."),
    thread_id: str = typer.Option(None, "--thread-id", "-t", help="Resume an existing thread."),
    max_loops: int = typer.Option(None, "--max-loops", "-n", help="Override max retrieval loops."),
    debug: bool = typer.Option(False, "--debug", "-d", help="Print full JSON trace after answer."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON-Lines output (answer + trace)."),
) -> None:
    """Ask a question and get a deterministic cited answer from the agent (F23).

    Unlike the interactive ``chat`` command this command prints exactly one
    answer and exits, making it suitable for scripting and CI pipelines.

    \b
    Examples::

        cce answer "Where is the login endpoint?"
        cce answer "List all CLI commands" --json
        cce answer "Explain the auth flow" --debug --max-loops 4
    """
    import json as _json  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    try:
        from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
        from cce.agents.graph import build_graph  # noqa: PLC0415
    except ImportError as e:
        console.print(f"[red]Agent dependencies missing: {e}[/]")
        raise typer.Exit(1)

    cfg = get_settings()
    if max_loops is not None:
        cfg.agent.max_retrieval_loops = max_loops

    tid = thread_id or str(uuid.uuid4())
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": tid}}
    state = {"query": query, "messages": [], "loop_count": 0,
             "retrieved_context": [], "reasoning_steps": [], "tool_errors": 0,
             "coverage_axes": {}}

    if not json_out:
        console.print(f"\n[bold cyan]Query:[/] {query}")
        with console.status("[bold green]Thinking…"):
            final_state = graph.invoke(state, config=config)
    else:
        final_state = graph.invoke(state, config=config)

    answer_text = final_state.get("answer", "")
    citations = final_state.get("citations", [])
    reasoning = final_state.get("reasoning_steps", [])

    if json_out:
        output = {
            "thread_id": tid,
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "reasoning_steps": reasoning,
        }
        print(_json.dumps(output))
        return

    # Human-readable output
    console.print()
    console.rule("[bold green]Answer")
    console.print(answer_text or "[dim]No answer generated.[/]")

    if citations:
        console.print()
        console.rule("[dim]Citations")
        for c in citations:
            console.print(f"  [cyan]{c}[/]")

    if debug:
        console.print()
        console.rule("[dim]Reasoning trace")
        for step in reasoning:
            console.print(f"  [dim]{step}[/]")
        console.print()
        console.print(f"[dim]thread_id={tid}  loops={final_state.get('loop_count', 0)}[/]")


if __name__ == "__main__":
    app()
