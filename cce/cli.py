"""Typer CLI entrypoint — all commands through Phase 5 are implemented."""

from __future__ import annotations

from pathlib import Path

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
) -> None:
    """Build index layers for the given codebase root.

    Add 'semantic' to also embed with nomic-embed-code into Qdrant (requires ~7GB VRAM).
    """
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    layer_list = [la.strip() for la in layers.split(",")]
    with console.status(f"Indexing {path} …"):
        stats = IndexPipeline().run(path, layer_list)
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
    """List all symbols in a file or directory."""
    from cce.indexer import IndexPipeline  # noqa: PLC0415

    sym_store = IndexPipeline().symbol_store
    rel = str(path)
    nodes = sym_store.get_for_file(rel)
    if kind:
        nodes = [n for n in nodes if n.kind.value.lower() == kind.lower()]
    tbl = Table(title=f"Symbols in {rel}", show_lines=False)
    tbl.add_column("Kind", style="cyan", width=12)
    tbl.add_column("Name", style="bold")
    tbl.add_column("Line", justify="right")
    tbl.add_column("Signature")
    for n in nodes:
        tbl.add_row(n.kind.value, n.name, str(n.line_start), (n.signature or "")[:60])
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

    # tree-sitter
    for lang in ("python", "typescript", "tsx"):
        try:
            from tree_sitter_languages import get_parser as _gp  # noqa: PLC0415
            _gp(lang); _ok(f"tree-sitter {lang}")
        except Exception as e:
            _fail(f"tree-sitter {lang}", str(e)[:80])

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
        cfg = _gs(); cfg.paths.qdrant_path.mkdir(parents=True, exist_ok=True)
        _QC(path=str(cfg.paths.qdrant_path)); _ok("Qdrant (embedded)", str(cfg.paths.qdrant_path))
    except Exception as e:
        _fail("Qdrant", str(e)[:80])

    # Embedder backend
    try:
        from cce.config import get_settings as _gs  # noqa: PLC0415
        from cce.embeddings.embedder import _resolve  # noqa: PLC0415
        cfg = _gs().embedder; model, dim = _resolve(cfg.backend, cfg.model_name, cfg.dim)
        if cfg.backend == "jina":
            importlib.import_module("sentence_transformers")
            _ok(f"Embedder ({cfg.backend})", f"{model}  dim={dim}  CPU-ready")
        elif cfg.backend == "openai":
            if os.getenv("OPENAI_API_KEY"):
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

    # API keys
    for var, note in [
        ("OPENAI_API_KEY", "needed for openai embedder + agent"),
        ("ANTHROPIC_API_KEY", "needed if CCE_AGENT__LLM_PROVIDER=anthropic"),
    ]:
        (_ok if os.getenv(var) else _warn)(var, "set" if os.getenv(var) else note)

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
    mcp: bool = typer.Option(False, help="Expose tools over Model Context Protocol."),
) -> None:
    """Start the FastAPI (and optional MCP) server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cce.server.app:create_app",
        host=host or settings.server.host,
        port=port or settings.server.port,
        factory=True,
        reload=False,
    )


@app.command()
def info() -> None:
    """Print resolved configuration."""
    settings = get_settings()
    console.print_json(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
