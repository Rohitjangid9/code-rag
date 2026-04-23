"""Central configuration loaded from env vars and optional `.cce.toml`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsSettings(BaseModel):
    """Filesystem locations used by the engine."""

    data_dir: Path = Field(default=Path(".cce"), description="Root for all indexes/DBs.")
    sqlite_db: Path = Field(default=Path(".cce/index.sqlite"))
    graph_db: Path = Field(default=Path(".cce/graph.kuzu"))
    qdrant_path: Path = Field(default=Path(".cce/qdrant"))
    agent_checkpoint: Path = Field(default=Path(".cce/agent.sqlite"))

    def resolve(self, repo_root: Path) -> "PathsSettings":
        """Return a copy with every *relative* path anchored to *repo_root*.

        Paths already absolute are left untouched so tests and overrides that
        set an explicit path (``settings.paths.sqlite_db = tmp_path / "x"``)
        continue to work as-is.
        """
        root = repo_root.resolve()

        def _anchor(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        return PathsSettings(
            data_dir=_anchor(self.data_dir),
            sqlite_db=_anchor(self.sqlite_db),
            graph_db=_anchor(self.graph_db),
            qdrant_path=_anchor(self.qdrant_path),
            agent_checkpoint=_anchor(self.agent_checkpoint),
        )


class EmbedderSettings(BaseModel):
    """Embedding backend configuration.

    Switch backends via .env:
        CCE_EMBED__BACKEND=openai        # default — best quality, needs OPENAI_API_KEY
        CCE_EMBED__BACKEND=jina          # CPU-friendly, no API key
        CCE_EMBED__BACKEND=nomic         # needs ~7GB VRAM
    """

    backend: Literal["nomic", "jina", "openai"] = "openai"
    # Model name — auto-detected dim when left at "" + backend default
    model_name: str = ""          # empty → per-backend sensible default
    dim: int = 0                  # 0 → auto-resolved from MODEL_DIMS lookup
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    dtype: Literal["fp16", "bf16", "fp32", "int8"] = "fp16"
    batch_size: int = 32          # larger default suits Jina on CPU
    max_tokens: int = 8192        # Jina v2 supports 8k context
    # Chunker budgets (word-level approximation, ~1.3 words/token)
    max_header_words: int = 100
    max_body_words: int = 1_400


class QdrantSettings(BaseModel):
    """Qdrant vector store configuration."""

    mode: Literal["embedded", "remote"] = "embedded"
    url: str | None = None
    api_key: str | None = None
    collection_prefix: str = "cce_"


class AgentSettings(BaseModel):
    """LangGraph agent runtime configuration."""

    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-5.4-mini-2026-03-17"
    llm_temperature: float = 0.0
    # F11: optional per-role model overrides; fall back to llm_model when None.
    planner_model: str | None = None
    responder_model: str | None = None
    max_retrieval_loops: int = 3
    checkpointer: Literal["memory", "sqlite"] = "sqlite"
    stream_events: bool = True
    debug: bool = False
    strict_citations: bool = True


class RetrievalSettings(BaseModel):
    """Retrieval pipeline configuration (F19/F20)."""

    # F19: synonym expansion — expand query terms before lex + vector calls
    synonym_expansion: bool = True
    # F20: cross-encoder reranker (requires sentence-transformers)
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 50   # candidates fed to reranker before final top-k


class ServerSettings(BaseModel):
    """HTTP server configuration."""

    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    # F30: API key auth — empty list = no auth required (dev default)
    api_keys: list[str] = Field(default_factory=list)
    # F30: token-bucket rate limit per IP; 0 = unlimited
    rate_limit_rpm: int = 0


class IndexerSettings(BaseModel):
    """Indexer pipeline configuration (F28)."""

    # Worker threads for parallel per-file parsing; 0 = use os.cpu_count()
    workers: int = 4

    # ── Indexing diagnostics (all off by default) ─────────────────────────
    # Enable with .env entries, e.g.:
    #   CCE_INDEXER__VERBOSE=true
    #   CCE_INDEXER__LOG_FILE=.cce/indexer.log
    #   CCE_INDEXER__JEDI_DEBUG=true
    #   CCE_INDEXER__EDGE_DEBUG=true

    # Per-file summary line at INFO level: symbols / edges / jedi hits
    verbose: bool = False
    # Write all indexer logs to this file (in addition to console).
    # Path is relative to the repo root unless absolute.
    log_file: Path | None = None
    # Log every Jedi Script creation, call-site count, and goto() result
    jedi_debug: bool = False
    # Log per-file edge breakdown by kind (IMPORTS/CALLS/REFERENCES/…)
    edge_debug: bool = False


class StoreSettings(BaseModel):
    """Symbol / graph store backend selection (F34)."""

    graph_backend: Literal["sqlite", "duckdb", "postgres"] = "sqlite"
    # Connection DSN used only when graph_backend != "sqlite"
    graph_dsn: str = ""


class Settings(BaseSettings):
    """Top-level settings aggregating all subsections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CCE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    schema_version: int = 1
    # F37: when True, all network calls (embedder, LLM, OTel) are blocked
    offline: bool = False

    # F-M1: absolute path of the indexed repo this Settings is bound to.
    # ``None`` means "no anchor" — paths fall through to the process CWD.
    repo_root: Path | None = None

    paths: PathsSettings = Field(default_factory=PathsSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    indexer: IndexerSettings = Field(default_factory=IndexerSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)

    def ensure_dirs(self) -> None:
        """Create on-disk directories referenced by the config."""
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.qdrant_path.mkdir(parents=True, exist_ok=True)


# F-M1/F-M2: repo-keyed settings cache.  Key is the resolved repo root
# (or ``None`` when no anchor is detected — single-repo legacy mode).
_settings_cache: dict[Path | None, Settings] = {}


def _walk_up_find_cce(start: Path) -> Path | None:
    """Walk up from *start* looking for a ``.cce/index.json`` marker.

    Mirrors git's behaviour for locating the working-tree root.  Returns
    ``None`` if no marker is found before the filesystem root.
    """
    current = start.resolve()
    while True:
        if (current / ".cce" / "index.json").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _resolve_repo_root(explicit: Path | None) -> Path | None:
    """Resolve the repo root from (in order): explicit arg, env, walk-up."""
    if explicit is not None:
        return explicit.resolve()
    env_val = os.environ.get("CCE_REPO_ROOT")
    if env_val:
        return Path(env_val).resolve()
    return _walk_up_find_cce(Path.cwd())


def get_settings(repo_root: Path | None = None) -> Settings:
    """Return a Settings instance bound to *repo_root* (or the active repo).

    Resolution order when *repo_root* is None:
    1. ``CCE_REPO_ROOT`` environment variable
    2. Walk-up from CWD looking for ``.cce/index.json`` (git-style)
    3. No anchor — paths stay relative to CWD (legacy behaviour)
    """
    root = _resolve_repo_root(repo_root)
    if root not in _settings_cache:
        s = Settings()
        if root is not None:
            s.paths = s.paths.resolve(root)
            s.repo_root = root
        s.ensure_dirs()
        _settings_cache[root] = s
    return _settings_cache[root]


def reset_settings_cache() -> None:
    """Test helper: drop all cached Settings instances."""
    _settings_cache.clear()
