# Setup & Getting Started Guide

Complete walkthrough from a fresh clone to a working `cce` installation.

---

## 1 · Prerequisites

- **Python >= 3.12** (checked in `pyproject.toml`)
- **Git**
- **pip** (or `uv`)
- Optional: **Node.js** (for future ts-morph TypeScript resolver)

Verify Python version:

```bash
python --version   # should print 3.12+
```

---

## 2 · Create a virtual environment (recommended)

```bash
cd code-context-extractor

# venv
python -m venv .venv
# Windows PowerShell
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

---

## 3 · Install the package

```bash
# base install — enough for CLI, indexing, SQLite, BM25 search
pip install -e ".[dev]"

# + agent LLM providers (OpenAI, Anthropic, Ollama)
pip install -e ".[dev,agents]"

# + local GPU embeddings (torch, transformers, ~7 GB VRAM)
pip install -e ".[dev,agents,embeddings]"

# + Kùzu graph store (optional, replaces SQLite graph)
pip install -e ".[dev,graph]"
```

---

## 4 · Create the `.env` file

The project **does not ship** an `.env.example`. Create `.env` in the repo root and paste the blocks you need.

### Minimal `.env` (works out of the box, CPU only)

```ini
# ── Top-level ──────────────────────────
CCE_LOG_LEVEL=INFO
CCE_SCHEMA_VERSION=1

# ── Paths ──────────────────────────────
CCE_PATHS__DATA_DIR=.cce
CCE_PATHS__SQLITE_DB=.cce/index.sqlite
CCE_PATHS__GRAPH_DB=.cce/graph.kuzu
CCE_PATHS__QDRANT_PATH=.cce/qdrant
CCE_PATHS__AGENT_CHECKPOINT=.cce/agent.sqlite

# ── Embedder ───────────────────────────
# Jina v2 — CPU-friendly, Apache-2.0, ~550 MB one-time download
CCE_EMBED__BACKEND=jina
CCE_EMBED__MODEL_NAME=jinaai/jina-embeddings-v2-base-code
CCE_EMBED__DEVICE=cpu
CCE_EMBED__BATCH_SIZE=32

# ── Qdrant ─────────────────────────────
CCE_QDRANT__MODE=embedded

# ── Agent ──────────────────────────────
CCE_AGENT__LLM_PROVIDER=openai
CCE_AGENT__LLM_MODEL=gpt-4o-mini
CCE_AGENT__MAX_RETRIEVAL_LOOPS=3
CCE_AGENT__CHECKPOINTER=sqlite
CCE_AGENT__STREAM_EVENTS=true

# ── Server ─────────────────────────────
CCE_SERVER__HOST=127.0.0.1
CCE_SERVER__PORT=8765
CCE_SERVER__CORS_ORIGINS=["*"]
```

### Using OpenAI embeddings instead of Jina

Replace the embedder block:

```ini
CCE_EMBED__BACKEND=openai
CCE_EMBED__MODEL_NAME=text-embedding-3-large
CCE_EMBED__DIM=3072
OPENAI_API_KEY=sk-your-key-here
```

### Using Anthropic agent provider

Add or change:

```ini
CCE_AGENT__LLM_PROVIDER=anthropic
CCE_AGENT__LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Using local GPU embeddings (Nomic)

Replace the embedder block:

```ini
CCE_EMBED__BACKEND=nomic
CCE_EMBED__MODEL_NAME=nomic-ai/nomic-embed-code
CCE_EMBED__DIM=3584
CCE_EMBED__DEVICE=cuda
CCE_EMBED__DTYPE=fp16
```

> **Tip:** After changing `.env`, run `cce info` to confirm the new values were picked up.

---

## 5 · Health check

Run this **before** indexing anything. It validates tree-sitter, SQLite, Qdrant, embedder backend, GPU, API keys, and installed packages.

```bash
cce doctor
```

Expected output: a table with green checks. Yellow warnings are okay (e.g., no GPU). Red failures need fixing before proceeding.

---

## 6 · Verify configuration

```bash
cce info
```

Prints the resolved `Settings` JSON so you can double-check paths, backend names, and API keys.

---

## 7 · Run tests

```bash
pytest -q                              # all 12 phases
pytest tests/test_phase1_walker.py -v  # one phase
pytest -k "django" -v                  # filter by keyword
```

---

## 8 · Index your first repo

### Scan (dry run, no DB writes)

```bash
cce scan /path/to/project
```

### Full index (L1–L3 + framework detection)

```bash
cce index /path/to/project
```

### Add semantic / vector layer

```bash
cce index /path/to/project --layers lexical,symbols,graph,framework,semantic
```

Layer cheat-sheet:

| Layer | What it does |
|-------|--------------|
| `lexical` | BM25 full-text over raw source |
| `symbols` | AST nodes: functions, classes, routes, models |
| `graph` | Call, import, inheritance edges |
| `framework` | Django / FastAPI / React specific edges |
| `semantic` | Vector chunks in Qdrant (needs embedder) |

---

## 9 · Query the index

```bash
# Hybrid search (BM25 + vector + graph expansion)
cce query "how is auth middleware wired?"

# Pure lexical (fastest, always works)
cce query "User model" --mode lexical

# Look up one symbol
cce get app.users.views.UserViewSet

# Call graph
cce callers app.users.models.User.save

# N-hop neighborhood
cce neighborhood app.users.models.User --depth 2

# Framework route resolution
cce get-route "/api/v1/users/{user_id}"
cce get-api-flow "/api/v1/users/42"
```

---

## 10 · Start the HTTP + MCP server

```bash
cce serve
```

- Swagger UI: http://127.0.0.1:8765/docs
- MCP JSON-RPC: `POST http://127.0.0.1:8765/mcp`
- MCP tool list: `GET http://127.0.0.1:8765/mcp/tools`

---

## 11 · Keep the index fresh while coding

```bash
cce watch /path/to/project
```

Re-indexes any file you save. Press `Ctrl-C` to stop.

---

## 12 · Common troubleshooting

| Symptom | Fix |
|---------|-----|
| `cce` command not found | Ensure the venv is active and `pip install -e ".[dev]"` succeeded. |
| `ModuleNotFoundError` | Run `cce doctor` to see which package is missing, then `pip install -e ".[dev]"`. |
| `.env` changes ignored | `get_settings()` caches the first load. Restart the shell or run `python -c "import cce.config; cce.config._settings=None"`. |
| Qdrant / SQLite locks | Make sure only one `cce` process is running. Delete `.cce/` and re-index if corruption is suspected. |
| Semantic indexing fails | Check `cce doctor` embedder line. If using `openai`, verify `OPENAI_API_KEY`. If using `nomic`, ensure GPU has >= 7 GB VRAM. |

---

## Next steps

- Read [`architecture.md`](./architecture.md) for pipeline internals.
- Read [`cli-reference.md`](./cli-reference.md) for every flag.
- Read [`configuration.md`](./configuration.md) for the full env var catalog.
- Read [`api-reference.md`](./api-reference.md) for the HTTP / MCP surface.
