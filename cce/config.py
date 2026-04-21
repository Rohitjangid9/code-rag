"""Central configuration loaded from env vars and optional `.cce.toml`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsSettings(BaseSettings):
    """Filesystem locations used by the engine."""

    model_config = SettingsConfigDict(env_prefix="CCE_PATHS_", extra="ignore")

    data_dir: Path = Field(default=Path(".cce"), description="Root for all indexes/DBs.")
    sqlite_db: Path = Field(default=Path(".cce/index.sqlite"))
    graph_db: Path = Field(default=Path(".cce/graph.kuzu"))
    qdrant_path: Path = Field(default=Path(".cce/qdrant"))
    agent_checkpoint: Path = Field(default=Path(".cce/agent.sqlite"))


class EmbedderSettings(BaseSettings):
    """Embedding backend configuration.

    Switch backends via .env:
        CCE_EMBED__BACKEND=jina          # default — CPU-friendly, no API key
        CCE_EMBED__BACKEND=openai        # needs OPENAI_API_KEY
        CCE_EMBED__BACKEND=nomic         # needs ~7GB VRAM
    """

    model_config = SettingsConfigDict(env_prefix="CCE_EMBED__", extra="ignore")

    backend: Literal["nomic", "jina", "openai"] = "jina"
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


class QdrantSettings(BaseSettings):
    """Qdrant vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="CCE_QDRANT_", extra="ignore")

    mode: Literal["embedded", "remote"] = "embedded"
    url: str | None = None
    api_key: str | None = None
    collection_prefix: str = "cce_"


class AgentSettings(BaseSettings):
    """LangGraph agent runtime configuration."""

    model_config = SettingsConfigDict(env_prefix="CCE_AGENT_", extra="ignore")

    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    max_retrieval_loops: int = 3
    checkpointer: Literal["memory", "sqlite"] = "sqlite"
    stream_events: bool = True


class ServerSettings(BaseSettings):
    """HTTP server configuration."""

    model_config = SettingsConfigDict(env_prefix="CCE_SERVER_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseSettings):
    """Top-level settings aggregating all subsections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    schema_version: int = 1

    paths: PathsSettings = Field(default_factory=PathsSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    def ensure_dirs(self) -> None:
        """Create on-disk directories referenced by the config."""
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.qdrant_path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
