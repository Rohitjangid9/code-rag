# Configuration

Every setting is a Pydantic field on `cce.config.Settings` (loaded from
`.env`, then from process env vars). Nested groups use the delimiter `__`.
Copy `.env.example` to `.env` as a starting point.

```
.env.example  →  .env  →  cce.config.Settings  →  cce.config.get_settings()
```

The first call to `get_settings()` also creates the on-disk directories
(`data_dir`, `qdrant_path`).

## 1 · Paths — `CCE_PATHS_*`

| Env var | Default | Purpose |
|---------|---------|---------|
| `CCE_PATHS_DATA_DIR` | `.cce` | Root directory for every on-disk artifact. |
| `CCE_PATHS_SQLITE_DB` | `.cce/index.sqlite` | Holds `files`, `symbols`, `edges`, `chunks`, and FTS5 tables. |
| `CCE_PATHS_GRAPH_DB` | `.cce/graph.kuzu` | Reserved for the Kùzu graph store. |
| `CCE_PATHS_QDRANT_PATH` | `.cce/qdrant` | Embedded Qdrant storage directory. |
| `CCE_PATHS_AGENT_CHECKPOINT` | `.cce/agent.sqlite` | `SqliteSaver` checkpoint DB for LangGraph. |

These are paths relative to the current working directory by default. Pass
absolute paths if you run `cce` from multiple dirs but want one shared index.

## 2 · Embedder — `CCE_EMBED__*`

| Env var | Default | Values / Notes |
|---------|---------|----------------|
| `CCE_EMBED__BACKEND` | `openai` | `openai` (default, hosted), `jina` (CPU), `nomic` (GPU) |
| `CCE_EMBED__MODEL_NAME` | `""` | Empty → per-backend default. Override to pin a model. |
| `CCE_EMBED__DIM` | `0` | `0` → auto-resolved from the model. Override only if you know what you're doing. |
| `CCE_EMBED__DEVICE` | `auto` | `auto`, `cuda`, `cpu`, `mps` |
| `CCE_EMBED__DTYPE` | `fp16` | `fp16`, `bf16`, `fp32`, `int8` |
| `CCE_EMBED__BATCH_SIZE` | `32` | Increase for GPU, decrease for low-RAM CPU. |
| `CCE_EMBED__MAX_TOKENS` | `8192` | Jina v2 context size. |
| `CCE_EMBED__MAX_HEADER_WORDS` | `100` | Chunker header budget. |
| `CCE_EMBED__MAX_BODY_WORDS` | `1400` | Chunker body budget. |

### Backend recipes

```ini
# Default — OpenAI text-embedding-3-large, best hosted quality
CCE_EMBED__BACKEND=openai
CCE_EMBED__MODEL_NAME=text-embedding-3-large
CCE_EMBED__DIM=3072
OPENAI_API_KEY=sk-...

# Jina v2 code, Apache-2.0, ~550 MB download once, CPU-friendly
CCE_EMBED__BACKEND=jina
CCE_EMBED__MODEL_NAME=jinaai/jina-embeddings-v2-base-code

# nomic-embed-code (best offline quality, needs ~7 GB VRAM)
CCE_EMBED__BACKEND=nomic
CCE_EMBED__MODEL_NAME=nomic-ai/nomic-embed-code
CCE_EMBED__DIM=3584
CCE_EMBED__DEVICE=cuda
```

Any backend switch only affects **newly embedded chunks**; re-run
`cce index <path> --layers semantic` to reflect the change.

## 3 · Qdrant — `CCE_QDRANT_*`

| Env var | Default | Values |
|---------|---------|--------|
| `CCE_QDRANT__MODE` | `embedded` | `embedded` (local file under `CCE_PATHS_QDRANT_PATH`) or `remote` |
| `CCE_QDRANT__URL` | `None` | e.g. `http://localhost:6333` — required when `mode=remote` |
| `CCE_QDRANT__API_KEY` | `None` | Qdrant Cloud key (if applicable) |
| `CCE_QDRANT__COLLECTION_PREFIX` | `cce_` | Prefixed to every collection name. |

One collection is created per indexed repo (named from the repo root), which
keeps embeddings of different projects isolated even when sharing Qdrant.

## 4 · Agent — `CCE_AGENT_*`

| Env var | Default | Values |
|---------|---------|--------|
| `CCE_AGENT__LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `ollama` |
| `CCE_AGENT__LLM_MODEL` | `gpt-4o-mini` | Any model the provider supports. |
| `CCE_AGENT__LLM_TEMPERATURE` | `0.0` | `float` |
| `CCE_AGENT__MAX_RETRIEVAL_LOOPS` | `3` | Hard cap on the `planner ↔ reasoner` loop. |
| `CCE_AGENT__CHECKPOINTER` | `sqlite` | `memory` (in-process) or `sqlite` (persisted to `CCE_PATHS_AGENT_CHECKPOINT`). |
| `CCE_AGENT__STREAM_EVENTS` | `true` | Whether `/agent/query` streams intermediate events via SSE. |

Provider API keys are read from standard env vars:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. For Ollama, set the model name and
make sure the daemon is reachable at `OLLAMA_BASE_URL` (default
`http://localhost:11434`).

## 5 · Server — `CCE_SERVER_*`

| Env var | Default | Values |
|---------|---------|--------|
| `CCE_SERVER__HOST` | `127.0.0.1` | Bind address. |
| `CCE_SERVER__PORT` | `8765` | TCP port. |
| `CCE_SERVER__CORS_ORIGINS` | `["*"]` | JSON-encoded list of allowed origins. |

## 6 · Top-level

| Env var | Default | Values |
|---------|---------|--------|
| `CCE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CCE_SCHEMA_VERSION` | `1` | Bumping this invalidates indexes and forces a rebuild. |

## 7 · `.env` precedence

`pydantic-settings` applies values in this order (later wins):

1. Field defaults in `cce/config.py`
2. `.env` file in the CWD
3. Process environment variables
4. Values passed directly to `Settings(...)` (used only in tests)

`extra="ignore"` is set on every group, so unknown env vars are silently
skipped — handy during migration between names.

## 8 · Inspecting the resolved config

```bash
cce info
```

Emits the whole `Settings` object as formatted JSON, including which
backend/device/dim the embedder resolved to. Pair with `cce doctor` to
confirm dependencies line up with the configuration.

## 9 · Programmatic use

```python
from cce.config import get_settings

settings = get_settings()
settings.embedder.backend       # 'openai'
settings.paths.sqlite_db        # PosixPath('.cce/index.sqlite')
settings.agent.llm_model        # 'gpt-4o-mini'
```

`get_settings()` caches the first resolved instance; to re-read env vars
within a long-running process, clear the cache manually
(`cce.config._settings = None`).
