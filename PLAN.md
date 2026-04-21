# Code Context Engine — Full Build Plan

> Goal: Given **any path** to a codebase (Django, FastAPI, or React), index it into a
> 4-layer structure (Lexical → Symbol → Graph → Semantic) that a multi-agent system
> (planner / reasoner / retriever / coder) can query through a well-defined tool API,
> comparable in quality to Copilot, Cursor, Augment, Sourcegraph Cody, Aider, Continue.

---

## 1. Non-Negotiable Design Principles

1. **Hybrid over pure-vector.** Embeddings alone miss structural truth; we combine
   lexical (BM25), symbolic (AST), graph (references/calls), and semantic (vector).
2. **AST first, embeddings last.** Build cheap, deterministic layers before expensive ones.
3. **Framework-aware.** Django URL→View→Serializer→Model, FastAPI route→handler→schema,
   React component→props→hooks must be first-class graph edges, not generic symbols.
4. **Incremental.** A single file save re-indexes only that file + its dependents.
5. **Tool API, not raw search.** Agents call typed tools (`find_callers`, `get_route`)
   instead of dumping text into an LLM.
6. **Language-agnostic parsing.** Tree-sitter for uniformity + language-specific
   resolvers (Jedi / ts-morph) for type-accurate edges.
7. **SCIP-compatible schema** so components can be swapped later.

---

## 2. Final Architecture

```
                  ┌──────────────────────────────────────────┐
                  │       Agent Tool API (typed queries)     │
                  │  search_code / get_symbol / find_callers │
                  │  find_refs / get_route / get_component.. │
                  └──────────────────┬───────────────────────┘
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
 ┌─────────────┐            ┌──────────────────┐         ┌────────────────┐
 │  Retriever  │  ◀──RRF──▶ │  LangGraph       │ ◀─────▶ │ Graph Expander │
 │ (hybrid)    │            │  Agent Runtime   │         │   (1-2 hop)    │
 └──────┬──────┘            │ (planner/retr./  │         └────────┬───────┘
        │                   │  reasoner nodes) │                  │
        │                   └──────────────────┘                  │
        ▼                                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer 4 — Semantic   : Qdrant (code chunks + headers, nomic-embed-code)  │
│ Layer 3 — Graph      : Kùzu / SQLite (nodes, edges, SCIP-style)          │
│ Layer 2 — Symbols    : SQLite + FTS5 (functions/classes/routes/models)   │
│ Layer 1 — Lexical    : SQLite FTS5 + ripgrep fallback                    │
└──────────────────────────────────────────────────────────────────────────┘
                                     ▲
                                     │
                   ┌─────────────────┴─────────────────┐
                   │          Indexer Pipeline         │
                   │  Walker → Parser → Resolver →     │
                   │  Framework Extractor → Writer     │
                   └─────────────────┬─────────────────┘
                                     ▲
                                     │
                       ┌─────────────┴─────────────┐
                       │   File Watcher (watchdog) │
                       │   + Git diff (branch sw.) │
                       └───────────────────────────┘
```

---

## 3. Tech Stack (locked)

| Concern                 | Choice                                                  |
|-------------------------|---------------------------------------------------------|
| Language                | Python 3.12                                             |
| Parsing (all langs)     | **tree-sitter** + `tree-sitter-languages` bindings      |
| Python type resolution  | **Jedi** (fast) + optional **Pyright** via LSP fallback |
| JS/TS/React resolution  | **ts-morph** via Node sidecar process                   |
| Primary DB (L1+L2)      | **SQLite** + FTS5 (zero-ops, portable)                  |
| Graph DB (L3)           | **Kùzu** (embedded, Cypher, fast) — fallback SQLite CTE |
| Vector DB (L4)          | **Qdrant** (embedded mode or docker)                    |
| Embeddings (default)    | **nomic-embed-code** (local, 3584-dim, ~7GB VRAM)       |
| Embeddings (alt hosted) | `voyage-code-3` / `text-embedding-3-small` — pluggable  |
| Embeddings (alt local)  | `jina-embeddings-v2-base-code` — CPU fallback           |
| Reranker                | **Disabled for v1** (add later only if eval shows need) |
| Lexical                 | SQLite FTS5; ripgrep shelled-out for literal scans      |
| Watcher                 | **watchdog**                                            |
| Agent orchestration     | **LangChain + LangGraph** (stateful multi-agent graph)  |
| Interchange schema      | **SCIP** (Sourcegraph) emitted alongside internal DB    |
| CLI / API               | **Typer** (CLI) + **FastAPI** (HTTP tool server + MCP)  |
| Testing                 | `pytest` + fixture repos (mini Django/FastAPI/React)    |

---

## 4. Repository Layout (target)

```
code-context-extractor/
├── cce/                         # package root (rename from flat files)
│   ├── __init__.py
│   ├── cli.py                   # Typer entrypoint: `cce index <path>` etc.
│   ├── config.py                # Pydantic settings (paths, models, toggles)
│   ├── walker.py                # repo traversal + .gitignore respect
│   ├── hashing.py               # content hashing + change detection
│   ├── parsers/
│   │   ├── base.py              # Parser protocol
│   │   ├── tree_sitter_parser.py
│   │   ├── python_resolver.py   # Jedi-based reference resolution
│   │   └── ts_resolver/         # Node sidecar (ts-morph) + IPC client
│   ├── extractors/
│   │   ├── base.py
│   │   ├── generic_python.py    # classes/functions/imports
│   │   ├── generic_js.py        # components/hooks/imports
│   │   ├── django_extractor.py  # urls, models, views, serializers, signals
│   │   ├── fastapi_extractor.py # routes, pydantic, deps, routers
│   │   └── react_extractor.py   # JSX tree, props, hooks, routes
│   ├── graph/
│   │   ├── schema.py            # Node/Edge dataclasses (SCIP-aligned)
│   │   ├── kuzu_store.py
│   │   └── sqlite_store.py      # fallback
│   ├── index/
│   │   ├── symbol_store.py      # SQLite + FTS5
│   │   ├── lexical_store.py
│   │   └── vector_store.py      # Qdrant client
│   ├── embeddings/
│   │   ├── chunker.py           # semantic chunking + headers
│   │   └── embedder.py          # nomic-embed-code default; pluggable backends
│   ├── retrieval/
│   │   ├── hybrid.py            # BM25 + vector + RRF
│   │   ├── graph_expand.py      # N-hop neighborhood
│   │   └── tools.py             # typed agent tools (shared by CLI/API/agents)
│   ├── agents/                  # LangChain + LangGraph multi-agent runtime
│   │   ├── state.py             # AgentState (TypedDict)
│   │   ├── nodes.py             # planner / retriever / reasoner node fns
│   │   ├── graph.py             # LangGraph StateGraph assembly
│   │   └── tools.py             # LangChain Tool wrappers over retrieval.tools
│   ├── watcher/
│   │   ├── fs_watcher.py
│   │   └── git_watcher.py
│   ├── server/
│   │   ├── app.py               # FastAPI app factory + lifespan
│   │   ├── deps.py              # dependency injection (stores, embedder)
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── index.py         # POST /index, GET /index/status
│   │   │   ├── search.py        # POST /search (hybrid/lexical/semantic)
│   │   │   ├── symbols.py       # GET /symbols, /callers, /refs
│   │   │   ├── graph.py         # GET /neighborhood, /route, /component
│   │   │   └── agent.py         # POST /agent/query → LangGraph invoke
│   │   └── schemas/             # Pydantic request/response models
│   └── scip/
│       └── emitter.py           # export to SCIP protobuf
├── tests/
│   ├── fixtures/
│   │   ├── sample_django/
│   │   ├── sample_fastapi/
│   │   └── sample_react/
│   └── test_*.py
├── PLAN.md                      # this file
├── ARCHITECTURE.md              # existing
├── README.md
└── pyproject.toml
```

---

## 5. Canonical Data Schema (SCIP-aligned, simplified)

### 5.1 Node

| Field            | Type       | Notes                                              |
|------------------|------------|----------------------------------------------------|
| `id`             | str (ULID) | primary key                                        |
| `kind`           | enum       | File, Module, Class, Function, Method, Variable, Route, Model, Component, Hook, Serializer, PydanticModel, Middleware, Signal, URLPattern |
| `qualified_name` | str        | `app.module.Class.method`                          |
| `name`           | str        | short name                                         |
| `file_path`      | str        | repo-relative                                      |
| `line_start`     | int        |                                                    |
| `line_end`       | int        |                                                    |
| `signature`      | str?       | def/class signature                                |
| `docstring`      | str?       |                                                    |
| `language`       | enum       | python, typescript, javascript, tsx, jsx           |
| `framework_tag`  | enum?      | django, fastapi, drf, react                        |
| `visibility`     | enum       | public/private                                     |
| `content_hash`   | str        | for incremental updates                            |
| `meta`           | json       | framework-specific payload (e.g. `http_methods`)   |

### 5.2 Edge

| Field          | Type  | Notes                                                 |
|----------------|-------|-------------------------------------------------------|
| `src_id`       | str   |                                                       |
| `dst_id`       | str   |                                                       |
| `kind`         | enum  | IMPORTS, CALLS, INHERITS, DECORATES, REFERENCES, RETURNS_TYPE, PARAM_TYPE, RAISES, USES_MODEL, ROUTES_TO, RENDERS, USES_HOOK, USES_PROP, HANDLES_SIGNAL, DEPENDS_ON, MOUNTS_ROUTER |
| `location`     | json  | `{file, line, col}` of the edge occurrence            |
| `confidence`   | float | 1.0 for resolver-proven, <1 for heuristic             |

### 5.3 Chunk (vector layer)

| Field         | Type   | Notes                                                |
|---------------|--------|------------------------------------------------------|
| `chunk_id`    | str    |                                                      |
| `node_id`     | str    | FK to symbol node                                    |
| `header`      | str    | `path \| qname \| imports \| docstring` — 2-4 lines  |
| `body`        | str    | function/class/component source                      |
| `embedding`   | vec    | dim from chosen model                                |
| `token_count` | int    |                                                      |

---

## 6. Build Phases (execution order — each phase must be shippable)

### Phase 0 — Scaffolding (0.5 day)
- Convert flat scripts into `cce/` package.
- Add `pyproject.toml`, `Typer` CLI skeleton: `cce index <path>`, `cce query "..."`.
- Add `config.py` (Pydantic-settings, `.cce.toml` + env vars).
- Set up `pytest` with fixture mini-repos under `tests/fixtures/`.
- **Deliverable:** `cce --help` works; CI runs empty test suite.

### Phase 1 — Walker + Hashing + Language Detection (0.5 day)
- Respect `.gitignore` via `pathspec`.
- Skip binaries, `node_modules`, `__pycache__`, `.venv`, `dist`, `build`.
- Detect language from extension + shebang.
- SHA-256 per file stored in SQLite `files` table `(path, hash, mtime, lang)`.
- **Deliverable:** `cce scan <path>` prints file inventory grouped by language.

### Phase 2 — Layer 1: Lexical Index (0.5 day)
- SQLite FTS5 table `lex(path, content)` using porter + trigram tokenizer.
- `search_lexical(query, k)` returns `(path, line, snippet)`.
- Shell-out to **ripgrep** for regex-literal queries (fast path, no index needed).
- **Deliverable:** `cce search --lexical "TODO"` returns hits ranked by BM25.

### Phase 3 — Layer 2: Symbol Index via Tree-sitter (2 days)
- Integrate **tree-sitter** with queries (`.scm` files) for each language:
  - Python: `class_definition`, `function_definition`, `decorated_definition`, `import_statement`, `import_from_statement`.
  - TS/JS/TSX: `class_declaration`, `function_declaration`, `arrow_function` (with capitalized name = component heuristic), `import_statement`, `call_expression` for hooks.
- Emit `Node` rows to SQLite `symbols` table + FTS5 index on `(qualified_name, name, docstring, signature)`.
- Preserve line ranges and content hashes per symbol for incremental diff.
- **Deliverable:** `cce symbols <path>` lists every class/function/component; `cce get <qname>` returns signature + docstring.

### Phase 4 — Reference Resolution (2 days)
- **Python**: use **Jedi**'s `Script.get_references()` and `infer()` per identifier; emit `CALLS`, `REFERENCES`, `INHERITS`, `IMPORTS`, `RETURNS_TYPE`, `PARAM_TYPE` edges.
- **TS/React**: spawn a **Node sidecar** using `ts-morph`; expose JSON-RPC endpoints `findReferences`, `getType`, `getJsxChildren`. Python talks to it via stdio.
- Fallback heuristic for unresolved names: lower `confidence` on the edge so reranker can down-weight.
- **Deliverable:** `cce callers <qname>` and `cce refs <qname>` return precise lists.

### Phase 5 — Layer 3: Graph Store (1 day)
- Bootstrap **Kùzu** with node + edge schemas from §5.
- Bulk-load from SQLite symbols/edges tables.
- Implement `graph_expand(node_id, depth=2, edge_kinds=[...])` using Cypher.
- Keep a SQLite-CTE fallback behind the same interface (`GraphStore` protocol) for zero-dependency mode.
- **Deliverable:** `cce neighborhood <qname> --depth 2` prints a subgraph.

### Phase 6 — Framework Extractors (3 days, parallelizable)

**6a. Django** (1 day)
- Parse `urls.py` recursively (`path`, `re_path`, `include`) → emit `URLPattern` nodes + `ROUTES_TO` edges to view callables/classes.
- Parse `models.Model` subclasses → `Model` nodes with field list in `meta`.
- DRF: detect `serializers.ModelSerializer`/`Serializer` subclasses, link `USES_MODEL` to their `Meta.model`; link ViewSets→Serializers via `serializer_class`.
- `settings.MIDDLEWARE` list → `Middleware` nodes ordered.
- `@receiver(signal)` → `HANDLES_SIGNAL` edges.
- Admin registrations (`admin.site.register(Model, Admin)`).

**6b. FastAPI** (1 day)
- Detect `FastAPI()` and `APIRouter()` instances; follow variable bindings.
- Decorators `@app.get/post/...` and `@router.*` → `Route` nodes with `path`, `methods`, `status_code`, `tags` in `meta`; `ROUTES_TO` → handler function.
- `app.include_router(r, prefix=...)` → `MOUNTS_ROUTER` edge (track prefix composition).
- `Depends(f)` in signatures → `DEPENDS_ON` edges.
- `response_model=` / `BaseModel` subclasses → `PydanticModel` nodes with field types.

**6c. React** (1 day)
- Component detection: exported function/arrow with PascalCase returning JSX (via tree-sitter TSX query).
- `RENDERS` edges: for each JSX element `<Foo ...>` inside component body, link to resolved `Foo` (ts-morph resolves import).
- `USES_HOOK`: call expressions starting with `use[A-Z]` inside a component.
- `USES_PROP`: destructured params → prop names stored in `meta`.
- Routes: detect React Router `createBrowserRouter`, `<Route path=... element={<X/>}>` → `Route` nodes.
- Optional cross-stack edge: if a component calls `fetch('/api/...')` or `axios.get('/api/...')` and a Django/FastAPI route matches that path, emit a synthetic `CALLS_API` edge. **This is the killer feature.**

- **Deliverable:** indexing a Django + React mono-repo produces a single graph where a React button → axios call → FastAPI route → handler → Pydantic model → ORM model is one connected path.

### Phase 7 — Layer 4: Semantic / Vector Index (1.5 days)
- **Chunker** (`embeddings/chunker.py`):
  - One chunk per function/method/class/component (`node_id` = FK).
  - Prepend a **header** block:
    ```
    # path: app/users/views.py
    # symbol: app.users.views.UserViewSet.retrieve
    # framework: drf
    # imports: User, UserSerializer
    # docstring: Returns a single user by id.
    ```
  - Split oversized nodes (>1500 tokens) on logical boundaries (inner classes/methods, JSX subtrees) instead of fixed windows.
- **Embedder**: abstract `Embedder` interface; default impl `NomicEmbedCode` (loads `nomic-ai/nomic-embed-code` via `transformers`, 3584-dim, ~7GB VRAM, fp16 on GPU / int8 fallback on CPU). Additional back-ends (`VoyageEmbedder`, `OpenAIEmbedder`, `JinaLocalEmbedder`) live behind the same interface for A/B testing.
- **Qdrant** collection per repo; payload = `{node_id, path, qname, kind, framework_tag}` for filtered search. Vector size = 3584 for default embedder.
- Batch embed on first index (batch size auto-tuned to available VRAM); re-embed only chunks whose `content_hash` changed.
- **Deliverable:** `cce search --semantic "where do we validate JWT?"` returns top-k chunks with headers.

### Phase 8 — Hybrid Retriever (1 day) — no reranker in v1
- `retrieval/hybrid.py`:
  1. Run lexical BM25 (top 50) and vector (top 50) **in parallel** (`asyncio.gather`).
  2. Merge via **Reciprocal Rank Fusion** (`k=60`).
  3. Graph-expand top 15 results by 1 hop along `CALLS`, `INHERITS`, `ROUTES_TO`, `RENDERS` using the graph store.
  4. Return top-10 after dedupe by `(file, line_range)`.
- Each result carries `score`, `provenance` (`lex|vec|graph`), and the chunk header.
- **Reranker deliberately excluded in v1.** Phase 12 eval will decide if it's needed (add as Phase 8b if MRR < 0.6).
- **Deliverable:** `cce query "how is auth middleware wired?"` returns a ranked, deduplicated, context-rich result set.

### Phase 9 — Agent Tool API + LangGraph Runtime (2 days)

**9a. Typed tool surface** (`retrieval/tools.py`) — single source of truth, called by CLI, HTTP, and agents:
```python
search_code(query: str, mode: Literal["auto","lexical","semantic","hybrid"]="auto", k: int=10, filters: dict|None=None) -> list[Hit]
get_symbol(qualified_name: str) -> Symbol
get_file_outline(path: str) -> list[Symbol]
find_references(qualified_name: str) -> list[Location]
find_callers(qualified_name: str) -> list[Symbol]
find_implementations(qualified_name: str) -> list[Symbol]
get_neighborhood(qualified_name: str, depth: int=2, edge_kinds: list[str]|None=None) -> SubGraph
get_route(pattern_or_path: str) -> RouteInfo       # Django/FastAPI
get_component_tree(component_name: str) -> ComponentTree  # React
get_api_flow(route_or_component: str) -> CrossStackFlow   # uses CALLS_API edges
```

**9b. LangChain Tool wrappers** (`agents/tools.py`) — wrap each function with `@tool` decorator exposing JSON schemas derived from Pydantic models.

**9c. LangGraph multi-agent runtime** (`agents/graph.py`):
- `AgentState` (TypedDict): `query`, `plan`, `retrieved_context`, `reasoning_steps`, `answer`, `messages`.
- Nodes:
  - `planner` — decomposes the query into retrieval sub-tasks (which tools to call).
  - `retriever` — executes tool calls (lexical/semantic/graph) concurrently.
  - `reasoner` — inspects retrieved context, decides if more retrieval is needed (loop back to planner) or answer is ready.
  - `responder` — formats final answer with citations (file:line).
- Edges: `START → planner → retriever → reasoner → (planner | responder) → END`.
- Checkpointer: `MemorySaver` in-process for v1; `SqliteSaver` for persistence.

**9d. Server exposure** (`server/`):
- `FastAPI` app factory with lifespan (loads embedder, opens DB connections).
- Routes: `/index`, `/search`, `/symbols/*`, `/graph/*`, `/agent/query` (streams LangGraph events via SSE).
- Auto OpenAPI schema for all tools.
- MCP server mode: `cce serve --mcp` exposes the same tools over Model Context Protocol for Claude/Cursor/Continue.

- **Deliverable:** `POST /agent/query {"query": "how is auth wired?"}` streams back a reasoned answer with code citations; `curl localhost:8765/symbols/callers?qname=...` works for direct tool access.

### Phase 10 — Incremental Updates (1 day)
- `watchdog` observer on the indexed root.
- On file modify:
  1. Recompute hash; skip if unchanged.
  2. Delete all nodes + edges + chunks owned by that file.
  3. Re-run pipeline for the single file.
  4. Re-resolve edges pointing **into** that file from dirty-neighborhood files (reverse-import table).
- On git branch switch: `git diff --name-only` → batch re-index.
- Schema version stamp in DB; bump forces full rebuild.
- **Deliverable:** editing a function updates its embedding + edges within ~300 ms.

### Phase 11 — SCIP Export (0.5 day)
- Implement `scip/emitter.py` writing the official SCIP protobuf.
- Allows: (a) interop with Sourcegraph / `scip-cli`, (b) swapping our resolver for `scip-python` / `scip-typescript` later.
- **Deliverable:** `cce export --scip out.scip` produces a file readable by `scip print`.

### Phase 12 — Evaluation Harness (1 day)
- Build a small gold-set of queries per fixture repo (20 queries × 3 repos).
- Metrics: Recall@10, MRR, Context Precision (are the returned chunks actually needed to answer?).
- Compare four modes: lexical-only, vector-only, hybrid, hybrid+rerank+graph.
- Inspired by Greptile / Morph retrieval evals and SWE-bench-style context scoring.
- **Deliverable:** `cce eval` prints a table; CI fails if Recall@10 drops > 5%.

---

## 7. Framework-Specific Extraction Cheatsheet

### Django / DRF
| Source                                | Emit                                         |
|---------------------------------------|----------------------------------------------|
| `path('u/<int:id>', UserView.as_view())` | `URLPattern`→`ROUTES_TO`→`UserView`       |
| `class User(models.Model)`            | `Model` node + fields in `meta`              |
| `class S(ModelSerializer): Meta.model=User` | `Serializer`→`USES_MODEL`→`User`       |
| `@receiver(post_save, sender=User)`   | `HANDLES_SIGNAL` edge                        |
| `MIDDLEWARE=[...]`                    | ordered `Middleware` nodes                   |
| `admin.site.register(User, UserAdmin)`| `REGISTERS_ADMIN` edge                       |

### FastAPI
| Source                                | Emit                                         |
|---------------------------------------|----------------------------------------------|
| `@app.get('/u/{id}', response_model=U)` | `Route`(path, methods, response_model) + `ROUTES_TO` |
| `def handler(db = Depends(get_db))`   | `DEPENDS_ON` edge to `get_db`                |
| `app.include_router(r, prefix='/v1')` | `MOUNTS_ROUTER` (prefix tracked for effective path) |
| `class UserIn(BaseModel)`             | `PydanticModel` with field types             |

### React
| Source                                | Emit                                         |
|---------------------------------------|----------------------------------------------|
| `export function UserCard({id})`      | `Component` + prop list                      |
| `<UserCard id={x}/>` inside JSX       | `RENDERS` edge                               |
| `useAuth()`, `useUsers()` call        | `USES_HOOK` edge                             |
| `<Route path='/u' element={<U/>}/>`   | `Route`→`RENDERS`→component                  |
| `axios.get('/api/users')`             | candidate `CALLS_API` edge (resolved cross-stack) |

---

## 8. Retrieval Recipe (what the agent actually sees)

```
query ──► classify (identifier? question? error?)
        ├─ identifier  ──► symbol FTS  + graph expand (callers/refs)
        ├─ literal/err ──► ripgrep/lexical  + file outline
        └─ NL question ──► hybrid(BM25 + vector) → RRF → graph expand → rerank
                           └─► pack top-k chunks with headers + path breadcrumbs
```

Packing rule (Aider-style): always include the **file outline** of each hit before
the code body, and dedupe overlapping line ranges across chunks.

---

## 9. Performance Targets

| Operation                              | Target (100k LOC repo)       |
|----------------------------------------|------------------------------|
| Cold full index                        | ≤ 90 s (no embeddings) / ≤ 6 min (with embeddings) |
| Incremental file re-index              | ≤ 300 ms                     |
| `search_code` hybrid, k=10             | ≤ 400 ms p95                 |
| `find_callers`                         | ≤ 50 ms                      |
| `get_neighborhood` depth=2             | ≤ 120 ms                     |
| Disk footprint                         | ≤ 150 MB + embeddings        |

---

## 10. Reference Projects & Reading (study before building each phase)

| Source                                   | What to steal                                                           |
|------------------------------------------|-------------------------------------------------------------------------|
| **Sourcegraph SCIP spec** (`sourcegraph/scip`) | Node/edge schema, symbol naming scheme, cross-language uniformity. Base our §5 schema on this. |
| **`scip-python`, `scip-typescript`**     | Reference resolvers we can plug in as an alternative to Jedi/ts-morph.  |
| **Aider repo-map** (`paul-gauthier/aider`, `aider/repomap.py`) | Tree-sitter tag queries, PageRank-style ranking to fit max context into a token budget. |
| **Sourcegraph Cody** (blog: "How Cody understands your codebase") | Hybrid retrieval pipeline, graph context expansion, embeddings chunking strategy. |
| **Continue.dev** (`continuedev/continue`, `core/indexing/`) | Production-grade incremental indexer, chunker, and SQLite+LanceDB combo; MCP tool exposure. |
| **Greptile blog** ("How we built an AI that understands codebases") | Graph + embeddings hybrid, evaluation methodology, cross-file reasoning patterns. |
| **Morph blog** ("Retrieval for code agents") | Reranker choice, chunk header design, eval harness structure. |
| **Tree-sitter query docs**               | `.scm` query authoring for each language.                               |
| **Qdrant docs — hybrid search**          | Native BM25 + dense fusion if we consolidate on Qdrant later.           |
| **Kùzu docs — Cypher**                   | Graph query patterns for neighborhood expansion.                        |

---

## 11. Risks & Mitigations

| Risk                                          | Mitigation                                                        |
|-----------------------------------------------|-------------------------------------------------------------------|
| Dynamic Python imports confuse Jedi           | Accept lower confidence; fall back to name-based heuristic edge.  |
| `ts-morph` sidecar adds Node dependency       | Vendor a pinned Node 20 binary in `tools/`; feature-flag fallback to pure tree-sitter queries (lower accuracy). |
| Large monorepos blow memory                   | Stream per-file; batch writes; use Qdrant on-disk mode.           |
| nomic-embed-code VRAM (~7GB) on smaller GPUs  | Auto-detect VRAM; fall back to `jina-v2-base-code` (CPU-friendly) or int8 quantization. |
| Embedding throughput on big repos             | Batch + fp16; embed public/exported symbols first; re-embed only changed chunks. |
| Schema drift breaking old indexes             | `schema_version` in DB; auto-rebuild on bump.                     |
| Cross-stack `CALLS_API` false positives       | Require literal string match + path template match; mark confidence<1. |
| LangGraph state explosion on long sessions    | Cap retrieval loop iterations; use `SqliteSaver` checkpointer; truncate old messages. |

---

## 12. Milestone Summary (suggested order)

1. **Week 1** — Phases 0–3 (package, walker, lexical, symbols). Usable for symbol search.
2. **Week 2** — Phases 4–5 (references + graph). Usable for "find callers/refs".
3. **Week 3** — Phase 6 (Django + FastAPI + React extractors). Framework-aware graph.
4. **Week 4** — Phases 7–8 (nomic-embed-code + hybrid retriever). Full RAG, no reranker.
5. **Week 5** — Phase 9 (LangGraph agent runtime + FastAPI + MCP). Agent-ready.
6. **Week 6** — Phases 10–12 (watcher, SCIP export, eval harness). Productionize.

Ship after Week 2; everything after that compounds quality.

---

## 13. Locked Decisions (v1)

- **Embedder**: `nomic-embed-code` (local, 3584-dim). Pluggable backends for later A/B.
- **Reranker**: disabled in v1. Revisit only if Phase 12 eval shows MRR < 0.6.
- **Agent orchestration**: LangChain (tools) + LangGraph (state graph).
- **Qdrant**: embedded mode for dev; Docker for prod (same client code).
- **Graph DB**: Kùzu default; SQLite CTE fallback behind `GraphStore` protocol.
- **API surface**: FastAPI (HTTP + SSE) + MCP server — both share the same tool registry.
