# Architecture

`cce` is built around one invariant: **every consumer (CLI, HTTP server, MCP,
LangGraph agent) calls the same typed functions in `cce.retrieval.tools`.**
That module is the only public retrieval surface; everything above it is a
wrapper, everything below it is a store.

## 1 · High-level diagram

```
          ┌────────────────────────────────────────────────┐
          │  CLI (typer)    HTTP (FastAPI)    MCP (JSON-RPC)│
          └──────────────────────┬─────────────────────────┘
                                 │  same functions
                 ┌───────────────┴────────────────┐
                 │   cce.retrieval.tools          │
                 │   search_code / get_symbol /   │
                 │   find_callers / find_refs /   │
                 │   get_neighborhood /           │
                 │   get_route / get_component_   │
                 │   tree / get_api_flow          │
                 └───────────────┬────────────────┘
                                 │
       ┌─────────────────────────┼────────────────────────────┐
       ▼                         ▼                            ▼
┌────────────────┐       ┌────────────────┐         ┌──────────────────┐
│ HybridRetriever│◀─RRF─▶│ LangGraph      │◀──────▶ │ Graph expander   │
│ (lex+vec+graph)│       │ (planner/retr./│         │ (1–2 hop, typed  │
└───────┬────────┘       │  reasoner/resp)│         │  edge kinds)     │
        │                └────────────────┘         └────────┬─────────┘
        ▼                                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ L4 Semantic  : Qdrant (embedded or remote) — Jina/OpenAI/Nomic       │
│ L3 Graph     : SQLite graph store (Kùzu pluggable), SCIP-aligned     │
│ L2 Symbols   : SQLite + FTS5 — tree-sitter-extracted nodes           │
│ L1 Lexical   : SQLite FTS5 — raw source                              │
└──────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
                 ┌───────────────┴───────────────┐
                 │ IndexPipeline (cce.indexer)   │
                 │ walk → hash → lex → parse →   │
                 │ framework-extract → resolve → │
                 │ graph → (embed) → write       │
                 └───────────────┬───────────────┘
                                 ▲
                                 │
                 ┌───────────────┴───────────────┐
                 │ watcher/ — watchdog + git-diff│
                 └───────────────────────────────┘
```

## 2 · The indexing pipeline (`cce.indexer.IndexPipeline`)

Triggered by `cce index <path>` or `POST /index`. Each file passes through:

1. **Walk** — `cce.walker.walk_repo` traverses the tree, respecting
   `.gitignore` via `pathspec`, skipping `node_modules`, `__pycache__`,
   `.venv`, `dist`, `build`, binaries, and files > configurable size.
   Language is detected from extension + shebang.
2. **Hash + change-set** — `cce.hashing` computes a SHA-256 per file and
   compares to the `files` table in SQLite. Only `new + changed` files are
   re-processed; `deleted` files have their nodes, edges, and chunks removed.
3. **Lexical** — `cce.index.lexical_store.LexicalStore.upsert()` writes the
   source into an FTS5-indexed SQLite table (`porter + trigram` tokenizer).
4. **Parse (tree-sitter)** — `cce.parsers.tree_sitter_parser.TreeSitterParser`
   uses `tree-sitter-languages` bindings to emit `Node` rows
   (`Class`, `Function`, `Method`, `Component`, `Hook`, …) plus **raw edges**
   (unresolved qualified-name targets) for imports, inheritance, calls.
5. **Framework extraction** — if the repo was detected as Django, FastAPI,
   and/or React (`cce.extractors.framework_detector.detect_frameworks`), the
   corresponding extractor augments the node list with `URLPattern`, `Route`,
   `Model`, `Serializer`, `PydanticModel`, `Middleware`, `Signal`, and emits
   `ROUTES_TO`, `USES_MODEL`, `DEPENDS_ON`, `MOUNTS_ROUTER`, `RENDERS`,
   `USES_HOOK`, `HANDLES_SIGNAL`, `CALLS_API` raw edges.
6. **Reference resolution** — for Python, `cce.parsers.python_resolver` uses
   **Jedi** to resolve each identifier to its definition (precise `CALLS`,
   `REFERENCES`, `INHERITS`, `PARAM_TYPE`, `RETURNS_TYPE` edges). For JS/TS,
   `cce.parsers.js_resolver` does best-effort name-based resolution (a
   `ts-morph` Node sidecar is the planned upgrade).
7. **Graph write** — `cce.graph.sqlite_store.SQLiteGraphStore` dedupes edges
   by `(src_id, dst_id, kind, file, line)` and stores them with a
   `confidence` score (1.0 for resolver-proven, <1 for heuristic).
8. **(Optional) Semantic** — if `semantic` is in `--layers`:
   - `cce.embeddings.chunker.chunk_nodes` emits one chunk per
     function / method / class / component with a small **header** block
     (`path | qname | framework | imports | docstring`) prepended to the body.
   - `cce.embeddings.embedder.get_embedder()` resolves to OpenAI (default),
     OpenAI (hosted), or Nomic (local GPU), all behind the same interface.
   - Chunks are upserted to a per-repo **Qdrant collection** with payload
     `{node_id, path, qname, kind, framework_tag}` for filtered search.
   - Only chunks whose `content_hash` changed are re-embedded.

## 3 · The data schema (`cce.graph.schema`)

All three layers share one SCIP-aligned model. Full enum lists:

- **`NodeKind`** — `File, Module, Class, Function, Method, Variable, Route,
  Model, Component, Hook, Serializer, PydanticModel, Middleware, Signal,
  URLPattern`.
- **`EdgeKind`** — `IMPORTS, CALLS, INHERITS, DECORATES, REFERENCES,
  RETURNS_TYPE, PARAM_TYPE, RAISES, USES_MODEL, ROUTES_TO, RENDERS,
  USES_HOOK, USES_PROP, HANDLES_SIGNAL, DEPENDS_ON, MOUNTS_ROUTER, CALLS_API`.
- **`Language`** — `python, typescript, javascript, tsx, jsx`.
- **`FrameworkTag`** — `django, drf, fastapi, react`.

A `Node` carries `id` (ULID), `kind`, `qualified_name`, `name`, `file_path`,
`line_start/end`, optional `signature` + `docstring`, `language`,
`framework_tag`, `visibility`, `content_hash`, and a `meta` JSON blob used for
framework-specific payloads (e.g. `http_methods`, `props`, `fields`).

An `Edge` carries `src_id`, `dst_id`, `kind`, optional `location`
(`{file, line, col}`), and `confidence ∈ [0, 1]`.

A `Chunk` (L4) has `chunk_id`, FK `node_id`, `header`, `body`, `embedding`,
and `token_count`.

## 4 · Hybrid retrieval (`cce.retrieval.hybrid`)

`HybridRetriever.retrieve(query, k)` executes, in order:

1. Symbol FTS5 + file FTS5 + (optional) Qdrant vector search **in parallel**.
2. Merges with **Reciprocal Rank Fusion** (`k=60`).
3. Picks the top 15 and **graph-expands** each by 1 hop along
   `CALLS, INHERITS, ROUTES_TO, RENDERS`.
4. Dedupes by `(file, line_range)` and returns the top-`k` as `Hit` objects
   each carrying `score`, `provenance ∈ {lex, vec, graph, hybrid}`, and the
   chunk header.

No reranker is used in v1 — the Phase 12 eval harness decides whether one is
added later.

## 5 · Agent runtime (`cce.agents`)

`cce.agents.graph.get_agent_graph()` returns a compiled LangGraph app.

- **State** (`cce.agents.state.AgentState`, TypedDict): `query`, `plan`,
  `retrieved_context`, `reasoning_steps`, `answer`, `messages`.
- **Nodes**: `planner` → `retriever` → `reasoner` → (loop back to `planner`
  up to `CCE_AGENT_MAX_RETRIEVAL_LOOPS` times) → `responder`.
- **Tools**: the nine `@tool`-decorated functions in `cce.agents.tools`
  (`ALL_TOOLS`) — thin wrappers over `cce.retrieval.tools`, used by both the
  LangChain agent and the MCP endpoint.
- **Checkpointer**: `MemorySaver` or `SqliteSaver` (controlled by
  `CCE_AGENT__CHECKPOINTER`).

## 6 · Incremental updates (`cce.watcher`)

`cce watch <path>` runs a `watchdog` observer; on file save it debounces
(`--debounce`, default 1.0 s), recomputes the hash, and re-runs the pipeline
for that single file plus any reverse-import dependents. On a branch switch
the git watcher runs `git diff --name-only` and batches the change-set into a
single pipeline invocation. The DB carries a `schema_version` stamp; a bump
forces a full rebuild.
