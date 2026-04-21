# Code Context Engine — Quick Start

## 1 · Install

```bash
cd code-context-extractor
pip install -e ".[dev]"          # base — no GPU, no API key needed
pip install -e ".[dev,agents]"   # + OpenAI / Anthropic / Ollama support
```

Copy `.env.example` → `.env`. Defaults work out of the box with the Jina CPU embedder.

---

## 2 · Health check (run this first)

```bash
cce doctor
```

Shows ✓ / ⚠ / ✗ for tree-sitter, SQLite, Qdrant, embedder backend, GPU, API keys, packages.

---

## 3 · Index a repo

```bash
cce index /path/to/repo
# default layers: lexical + symbols + graph + framework

# also embed (needs CCE_EMBED__BACKEND configured in .env)
cce index /path/to/repo --layers "lexical,symbols,graph,framework,semantic"
```

| Layer | What it does |
|---|---|
| `lexical` | BM25 full-text search over source files |
| `symbols` | AST symbols — functions, classes, routes, models |
| `graph` | Call, inheritance, import edges |
| `framework` | Django / FastAPI routes + DRF serializers + React routes |
| `semantic` | Vector embeddings → Qdrant (Jina/OpenAI/Nomic) |

---

## 4 · Things you can query

```bash
# Hybrid search (BM25 + vector + graph expansion) — best results
cce query "how is authentication middleware wired?"

# Pure lexical (fastest, always works)
cce query "User model" --mode lexical

# Vector only (needs semantic layer indexed)
cce query "JWT validation" --mode semantic

# Look up one symbol by qualified name
cce get "app.users.views.UserViewSet"

# Explore the call graph around a symbol
cce neighborhood "app.users.models.User" --depth 2
```

---

## 5 · Framework-specific retrieval (after indexing)

```bash
# Resolve a URL to its handler + response model
cce get-route "/api/v1/users/{user_id}"

# Full UI → API → handler → model chain
cce get-api-flow "/api/v1/users/42"
```

---

## 6 · Watch for file changes (incremental re-index)

```bash
cce watch /path/to/repo          # re-indexes any saved file within ~1 s
```

---

## 7 · Start the API server

```bash
cce serve
# Swagger UI  →  http://127.0.0.1:8765/docs
# MCP endpoint → POST http://127.0.0.1:8765/mcp   (JSON-RPC 2.0)
# MCP tool list → GET http://127.0.0.1:8765/mcp/tools
```

---

## 8 · Export & Eval

```bash
# SCIP JSON (Sourcegraph-compatible symbol index)
cce export-scip /path/to/repo --out index.scip.json

# Retrieval quality report — MRR@10, Recall@10, nDCG@10
cce eval /path/to/repo --queries tests/fixtures/eval_queries.yaml
```

---

## 9 · Switch embedding backend

Edit `.env` — no code changes needed:

```bash
# Default: Jina v2 — CPU, Apache-2.0, ~550 MB download once
CCE_EMBED__BACKEND=jina

# OpenAI (best quality)
CCE_EMBED__BACKEND=openai
CCE_EMBED__MODEL_NAME=text-embedding-3-large
OPENAI_API_KEY=sk-...

# Nomic (best offline, needs ~7 GB VRAM)
CCE_EMBED__BACKEND=nomic
CCE_EMBED__DEVICE=cuda
```

---

## 10 · Run tests

```bash
pytest -q                              # all phases
pytest tests/test_phase8_hybrid.py -v  # one phase
pytest -k "django" -v                  # filter by name
```

---

## Repo layout

```
cce/
  cli.py          ← all commands (index, query, watch, serve, doctor, eval…)
  config.py       ← all settings — driven by .env / CCE_* env vars
  indexer.py      ← pipeline orchestrator (walk→parse→extract→embed)
  parsers/        ← tree-sitter AST → symbols + edges (Python, TS, JS)
  extractors/     ← Django / FastAPI / React framework symbols
  retrieval/      ← hybrid.py (BM25+vector+graph), tools.py (9 LangChain tools)
  agents/         ← LangGraph agent (planner→retriever→reasoner→responder)
  embeddings/     ← chunker + Jina / OpenAI / Nomic embedders
  index/          ← SQLite stores (lexical FTS5, symbols, vector→Qdrant)
  graph/          ← schema + SQLite graph store
  scip/           ← SCIP JSON export
  eval/           ← MRR / Recall / nDCG harness
  watcher/        ← watchdog file watcher + git-diff batch watcher
  server/         ← FastAPI app + MCP JSON-RPC 2.0 endpoint
tests/            ← one file per phase: test_phase1_*.py … test_phase12_*.py
PLAN.md           ← full design & decision log
.env.example      ← every configurable setting with comments
```