# Code Context Engine (`cce`) — Documentation

A local-first, 4-layer code indexer and agent runtime that turns any Django,
FastAPI, or React codebase into a set of typed queries that an LLM agent (or a
human at a CLI) can use to answer questions about the code.

> Package name on PyPI / in `pyproject.toml`: **`cce`** · Python ≥ 3.12 · MIT

---

## 1 · What it is

`cce` indexes a repository into **four complementary layers** and exposes them
through a single typed tool surface that is reused by the CLI, the HTTP API,
and the LangGraph agent:

| Layer | Store | What it holds |
|------:|-------|---------------|
| **L1 Lexical** | SQLite FTS5 (+ optional ripgrep fallback) | Full-text source lines for BM25 search |
| **L2 Symbols** | SQLite + FTS5 | Functions, classes, methods, components, routes, models — extracted via **tree-sitter** |
| **L3 Graph** | SQLite graph store (Kùzu planned, pluggable) | `CALLS`, `IMPORTS`, `INHERITS`, `ROUTES_TO`, `RENDERS`, `USES_HOOK`, `CALLS_API`, … edges |
| **L4 Semantic** | Qdrant (embedded or remote) | Per-symbol code chunks embedded with Jina / OpenAI / Nomic |

On top of those layers the engine adds:

- A **hybrid retriever** (BM25 + vector + graph 1-hop expansion, fused with
  Reciprocal Rank Fusion).
- A **LangGraph multi-agent runtime** (`planner → retriever → reasoner →
  responder`) that calls the tools to answer natural-language questions.
- A **FastAPI server** exposing the same tools as REST endpoints and as
  **MCP (Model Context Protocol)** JSON-RPC for Claude / Cursor / Continue.
- Framework-aware extractors for **Django / DRF**, **FastAPI**, and **React**
  that emit first-class graph edges (URL → view → serializer → model, route →
  handler → Pydantic model, component → hook → API call).
- **Incremental re-indexing** via `watchdog` (per-file edits) and `git diff`
  (branch switches).
- **SCIP export** for interop with Sourcegraph tooling.
- An **evaluation harness** measuring MRR@k, Recall@k, and nDCG@k against a
  YAML gold set.

---

## 2 · What you can do with it

After `pip install -e ".[dev]"` and `cce index /path/to/repo`:

| Goal | Command |
|------|---------|
| List every file by language | `cce scan <path>` |
| Build the full index | `cce index <path> --layers lexical,symbols,graph,framework[,semantic]` |
| Search with BM25 over symbols | `cce search "User model" --mode symbols` |
| Run the hybrid (BM25 + vector + graph) retriever | `cce query "how is auth middleware wired?"` |
| Look up a symbol by qualified name | `cce get app.users.views.UserViewSet` |
| List symbols in one file | `cce symbols app/users/views.py` |
| Find all callers of a symbol | `cce callers app.users.models.User.save` |
| Show the N-hop subgraph around a symbol | `cce neighborhood app.users.models.User --depth 2` |
| Resolve a URL to its handler + response model | `cce get-route "/api/v1/users/{user_id}"` |
| Show the UI → API → handler → model chain | `cce get-api-flow "/api/v1/users/42"` |
| Watch for file changes and re-index | `cce watch <path>` |
| Start HTTP + MCP server | `cce serve` |
| Export a Sourcegraph-compatible SCIP index | `cce export-scip <path> --out index.scip.json` |
| Run the retrieval eval harness | `cce eval <path> --queries queries.yaml` |
| Check that the environment is healthy | `cce doctor` |
| Print resolved configuration | `cce info` |

See [`cli-reference.md`](./cli-reference.md) for every flag of every command.

---

## 3 · Why this shape

- **AST first, embeddings last.** Cheap, deterministic layers are built before
  anything that needs a GPU; pure-vector search misses structural truth.
- **Framework-aware.** Django URL → View → Serializer → Model, FastAPI
  route → handler → schema, and React component → hook → API call are
  first-class graph edges — not generic symbols.
- **One tool surface.** `cce.retrieval.tools` is the only place where
  retrieval logic lives. CLI, FastAPI routes, LangChain tools, and the MCP
  endpoint all dispatch to the same functions.
- **SCIP-aligned schema.** Node / Edge shapes follow Sourcegraph's SCIP so
  resolvers (Jedi, ts-morph, scip-python, scip-typescript) can be swapped
  later without touching the rest of the engine.
- **Local-first, pluggable.** SQLite + embedded Qdrant work out of the box;
  Kùzu, remote Qdrant, and hosted embedders are drop-in replacements.

---

## 4 · Documentation map

| File | Read this when… |
|------|-----------------|
| [`architecture.md`](./architecture.md) | You want to understand the pipeline, the layers, and the data schema. |
| [`cli-reference.md`](./cli-reference.md) | You want a full reference of every `cce <command>` with flags. |
| [`api-reference.md`](./api-reference.md) | You need the HTTP endpoints (Swagger) and the MCP JSON-RPC surface. |
| [`configuration.md`](./configuration.md) | You need to know every `CCE_*` env var and how to switch embedders / LLMs / stores. |
| [`../QUICKSTART.md`](../QUICKSTART.md) | You just want the shortest path from install to first query. |
| [`../PLAN.md`](../PLAN.md) | You want the design log: build phases, locked decisions, reference projects. |

---

## 5 · Project layout

```
code-context-extractor/
├── cce/                   # installable package (entry point: `cce` Typer app)
│   ├── cli.py             # Typer commands (scan, index, search, symbols, get,
│   │                      #   callers, refs, neighborhood, query, watch,
│   │                      #   export-scip, eval, serve, doctor, info)
│   ├── config.py          # Pydantic Settings (env-driven, nested)
│   ├── walker.py          # repo traversal with .gitignore + language detection
│   ├── hashing.py         # SHA-256 per file + change-set computation
│   ├── indexer.py         # IndexPipeline orchestrating every phase
│   ├── parsers/           # tree-sitter + Jedi (Python) + JS resolver
│   ├── extractors/        # Django / FastAPI / React framework extractors
│   ├── index/             # SQLite lexical + symbol stores + Qdrant vector store
│   ├── graph/             # Node/Edge schema (SCIP-aligned), SQLite & Kùzu stores
│   ├── embeddings/        # chunker + Jina / OpenAI / Nomic embedders
│   ├── retrieval/         # hybrid.py (BM25 + vec + graph + RRF) + tools.py
│   ├── agents/            # LangGraph state graph + LangChain tool wrappers
│   ├── server/            # FastAPI app factory + MCP JSON-RPC endpoint
│   ├── watcher/           # watchdog file watcher + git-diff batch watcher
│   ├── scip/              # SCIP protobuf/JSON emitter
│   └── eval/              # dataset loader + harness + metrics
├── tests/                 # one `test_phaseN_*.py` per build phase + fixtures
├── docs/                  # ← you are here
├── PLAN.md                # full design document
├── QUICKSTART.md          # shortest path to first query
├── .env.example           # every configurable setting, commented
└── pyproject.toml
```
