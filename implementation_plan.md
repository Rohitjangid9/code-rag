# Making CCE Fully Dynamic — Multi-Repo Robustness Audit

> **Goal:** Transform CCE from a "works-on-this-one-repo" tool into a proper multi-repo indexing engine where you can `cce index /any/repo` and everything Just Works™.

I've done a code-by-code audit of every module. Below is the complete diagnosis organized by **severity**, followed by a **proposed solution architecture** and a **phased execution plan**.

---

## 🔴 Critical — Things That Actively Break On Another Repo

### 1. **Global singleton DB — only one repo can be indexed at a time**

The entire system assumes a single SQLite database. There is **no concept of "which repo am I querying?"** at runtime.

| File | Problem |
|---|---|
| [config.py](file:///d:/Learn/code-context-extractor/cce/config.py#L12-L19) | `PathsSettings` defaults are all relative paths (`.cce/index.sqlite`). These resolve relative to the **CWD**, not the indexed repo root. |
| [db.py](file:///d:/Learn/code-context-extractor/cce/index/db.py#L143-L151) | `_managers: dict[Path, DatabaseManager]` is a module-level global — works as a singleton cache, but the key is the config path, not the repo. |
| [tools.py](file:///d:/Learn/code-context-extractor/cce/retrieval/tools.py#L59-L75) | `_pipeline()` and `_hybrid_retriever()` are `@lru_cache(maxsize=1)` — **permanently locked to the first repo** that called them. A second repo gets the wrong DB. |
| [deps.py](file:///d:/Learn/code-context-extractor/cce/server/deps.py#L10-L13) | `settings_dep()` is `@lru_cache(maxsize=1)` — server-wide, can never switch repos. |

**Impact:** If you index repo A, then index repo B, all queries still hit repo A's database unless you restart the entire process and change the CWD.

---

### 2. **Paths are relative to CWD, not to the indexed repo root**

```python
# config.py — these are RELATIVE paths with no anchor
class PathsSettings(BaseModel):
    data_dir: Path = Path(".cce")
    sqlite_db: Path = Path(".cce/index.sqlite")
    graph_db: Path = Path(".cce/graph.kuzu")
    qdrant_path: Path = Path(".cce/qdrant")
    agent_checkpoint: Path = Path(".cce/agent.sqlite")
```

And in `.env`:
```
CCE_PATHS__DATA_DIR=.cce
CCE_PATHS__SQLITE_DB=.cce/index.sqlite
```

**What happens:** The `.cce/` folder always gets created relative to wherever the user runs the CLI from. If they run `cd /home/user && cce index /projects/myapp`, the index lives at `/home/user/.cce/` — completely detached from the repo. If they then run `cd /tmp && cce query "where is login?"`, it creates a **new empty** `.cce/` and finds nothing.

---

### 3. **No `repo_root` stored or passed through the query path**

The indexer accepts `root: Path` at `run()` time and uses it — but the information is never persisted in a way the query-time code can discover.

| Module | What it does | What it should do |
|---|---|---|
| `IndexPipeline.run(root)` | Uses `root` for walking files, resolving refs | ✅ OK at index time |
| `_pipeline()` in retrieval tools | Creates `IndexPipeline()` with no root | ❌ Doesn't know which repo |
| `get_file_slice()` | Reads from `lex_fts` content | ❌ No root context — can't read actual files |
| `VectorStore.collection_name_from_db()` | Hashes the DB path, not the repo | ❌ Two repos with same DB name = collision |
| `HybridRetriever._vector_search()` | Derives collection from `settings.paths.sqlite_db` | ❌ Wrong repo if config doesn't match |

---

### 4. **Eval dataset is hardcoded to CCE's own qualified names**

[core.yaml](file:///d:/Learn/code-context-extractor/cce/eval/datasets/core.yaml) contains:
```yaml
expected_symbols:
  - "cce.indexer.IndexPipeline.run"
expected_files:
  - "cce/indexer.py"
```

**Impact:** Running `cce eval-agent /path/to/django-app` will score 0% because the golden answers reference CCE's own symbols. The eval system has no way to generate repo-specific expected values.

---

## 🟠 Major — Things That Silently Produce Wrong Results

### 5. **`.env` is loaded from CWD, not from the repo root**

```python
# config.py
model_config = SettingsConfigDict(
    env_file=".env",     # ← relative to CWD
    ...
)

# llm.py
load_dotenv(".env")      # ← relative to CWD
```

If a user has a per-repo `.env` with different API keys or model configs, it only works if they `cd` into that repo first. If they use absolute paths from a different directory, the wrong `.env` is loaded.

---

### 6. **Qdrant collection naming is fragile across repos**

```python
# vector_store.py
def collection_name(self, root: Path) -> str:
    h = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
    return f"cce_{h}"

def collection_name_from_db(self, db_path: Path) -> str:
    h = hashlib.sha1(str(db_path.resolve()).encode()).hexdigest()[:12]
    return f"cce_{h}"
```

Two different naming schemes: one hashes the **repo root**, the other hashes the **DB path**. The indexer uses `collection_name(root)` but the retriever uses `collection_name_from_db(db_path)` — these will **never** match unless `root == db_path` (which is never true).

> [!CAUTION]
> This means **vector search is silently broken** at query time. The retriever looks for a collection name that doesn't exist, gracefully returns `[]`, and the system appears to work but only uses BM25.

---

### 7. **`get_file_slice()` and `grep_code()` read the FTS snapshot, not live files**

```python
# Both tools do:
row = conn.execute("SELECT content FROM lex_fts WHERE path = ?", (path,))
```

Consistent-snapshot semantics is intentional (avoids tool disagreement within
one query), but three concrete problems follow:

1. **Stale after edits** — if the user edits a file and doesn't re-index,
   every tool answers with outdated content. No freshness check.
2. **No repo-root anchor** — `path` must literally match the relative path
   stored at index time. A user passing an absolute path (or a path from a
   different CWD) gets a silent "not found".
3. **`grep_code()` rebuilds a regex/tokenizer on every call** and streams
   through every file's `content` column — O(repo-size) per grep. At 10 k
   files this is already 200 ms; on large monorepos it's the slowest tool.

---

### 8. **Agent system prompt contains no repo context**

[llm.py](file:///d:/Learn/code-context-extractor/cce/agents/llm.py#L19-L47) — the `SYSTEM_PROMPT` says:
```
You are Code Context Engine — an expert code analysis assistant.
You have access to tools that index and query a codebase.
```

It doesn't tell the LLM:
- **Which** codebase is indexed
- What **language** the codebase is in
- What **framework** it uses
- How many files/symbols exist

This causes the agent to hallucinate repo-specific assumptions. For a Django app, it might try `list_routes(framework="fastapi")` because that's all it was trained with.

---

### 9. **`_ALWAYS_SKIP` in walker is too aggressive for some repos**

[walker.py](file:///d:/Learn/code-context-extractor/cce/walker.py#L18-L30):
```python
_ALWAYS_SKIP = frozenset({
    ...
    "migrations",    # skips Django migrations — important for some queries!
    "vendor",        # skips Go vendor — may contain forked code
    "target",        # skips Rust/Maven build output
})
```

For a Django-heavy repo, `migrations/` might be critical for answering "what
database changes happened?" For a Go repo with vendored deps, `vendor/`
might contain forked code that needs indexing. **Fix:** demote these from
`_ALWAYS_SKIP` to a default `--skip` CLI flag the user can override per repo.

---

### 10. **Framework extractors are brittle substring matchers**

`framework_detector.detect_frameworks()` and `file_belongs_to()` decide
framework membership by doing things like `"FastAPI" in source` and
`"ModelSerializer" in source`. Real consequences:

- A docstring mentioning `"FastAPI-style decorators"` flips the file to
  FRAMEWORK=FASTAPI — and then the FastAPI extractor runs over it, producing
  no routes but wasting work.
- `"from fastapi" in txt` misses `"from fastapi.routing import APIRouter"` on
  a multi-line import (`\` line continuation).
- `"rest_framework"` substring tags any file that mentions DRF in a comment.

Plus `detect_frameworks()` calls `root.rglob("*.py")` **with no
`.gitignore` awareness** — on a 10 k-file repo that walks
`node_modules/`, `.venv/`, `dist/`, and build outputs before deciding
anything. On my box this is 8-15 s per call.

**Fix:** run detection against *already-walked* files (reuse `walk_repo`),
and use AST checks (`ast.parse` → look for `ImportFrom(module="fastapi")`)
instead of substring matching for the decision-making step.

---

### 11. **FastAPI / Django extractors miss major real-world patterns**

FastAPI extractor catches `@router.get("/path")` but misses:
- `app.add_api_route("/path", handler, methods=[...])` — common in large apps
- `@app.websocket("/ws")` — WebSocket routes
- `APIRouter(prefix="/v1")` inline prefix (already handled at include_router
  level but not when the APIRouter is constructed with prefix directly)
- Sub-app mounts: `app.mount("/admin", admin_app)`

Django extractor catches `urlpatterns = [path(...), …]` but misses:
- `include('myapp.urls')` — URL patterns from included apps **lose their
  prefix entirely**. Same class of bug as the FastAPI router-prefix bug
  you fixed in F4 — but Django includes are nested and recursive.
- DRF `router.register(r'users', UserViewSet)` → auto-generated `/users/`,
  `/users/{pk}/`, `/users/{pk}/{action}/` routes. None of these are
  emitted; the ViewSet methods are invisible to `list_routes()`.
- `@action(detail=True, methods=['post'])` on ViewSet methods (custom
  routes that DRF auto-wires).
- Admin site routes (`admin.site.register`).

**Impact:** a Django/DRF project answers "List all API endpoints" with a
tiny fraction of the real route table.

---

### 12. **No cross-stack glue — React ↔ FastAPI ↔ Django never links up**

For a full-stack repo the single most valuable query is *"trace this API
call from the React click through to the database"*. Today:

- React extractor sees the `fetch("/api/users")` call as a string literal.
- FastAPI extractor emits a Route with `path="/api/users"`.
- **There is no edge connecting the two.** The agent has to do string
  matching at query time, which is slow and fragile.

Same for Django: a React component calling `/api/accounts/login/` has no
explicit link to the `LoginView` Django class-based view.

**Fix:** new edge kind `CALLS_API` (already exists in
`EdgeKind`!) — not currently emitted. A second-pass resolver should:
1. Collect all string literals in JSX/TSX files that look like URL paths.
2. Match them against `effective_path` on every Route/URLPattern node.
3. Emit `CALLS_API` edges from the enclosing component symbol → the Route.

Enables `find_callers(route_qname)` to return the React components, and
`get_neighborhood(component_qname, edge_kinds=["CALLS_API"])` to trace the
request flow end-to-end.

---

### 13. **JS/TS has no symbol-to-symbol resolver**

Python uses Jedi to resolve names to definitions; `js_resolver.py` only
does tree-sitter structural extraction. So for any TypeScript or React
repo:

- `find_callers("UserTable")` returns `[]` even when 5 components import
  and use it — because no CALLS/REFERENCES edges were emitted.
- Import-aliased calls (`import { foo as bar } from './x'; bar()`) are
  invisible.
- Default-export vs named-export resolution is missing.

**Fix (cheapest):** call `tsc --noEmit --listFiles` or use the
`tree-sitter-typescript` queries plus a small import-table pass to
resolve the common 90 % case. Full `tsserver` integration is Phase 4+.

---

## 🟡 Minor — Design Debt That Affects Multi-Repo UX

### 14. Server routes have no concept of "which repo"

- [agent.py](file:///d:/Learn/code-context-extractor/cce/server/routes/agent.py) — `POST /agent/query` has no `repo` field
- [index.py](file:///d:/Learn/code-context-extractor/cce/server/routes/index.py) — `POST /index` has `path` but it's unused (returns 501)
- No `/repos` endpoint to list indexed repos

### 15. No index isolation — one SQLite DB for everything

If two repos have a file `config.py`, their symbols will collide in the
same `symbols` table. `file_path` stores relative paths, and
`_node_id_from_qname("app.views.home")` hashes to the same id regardless
of which repo produced it — so even with per-repo DBs, the id space is
**not repo-isolated**. If you ever merge two indexes, rows overwrite.

**Fix:** include `repo_root` (or a short repo slug) as a salt in
`_node_id_from_qname`. Centralized mode then becomes safe automatically.

### 16. `.env.example` has production-y values baked in

Real Qdrant URL, specific model version — should be generic placeholders.

---

## Proposed Architecture — Making CCE Multi-Repo Ready

### Core Design: **Per-Repo Data Directory**

```
myrepo/
├── .cce/                          # index lives INSIDE the repo
│   ├── index.sqlite               # symbols, edges, lexical
│   ├── qdrant/                    # embedded vectors (if local)
│   ├── agent.sqlite               # checkpointer
│   └── index.json                 # manifest (already exists)
├── src/
├── tests/
└── ...
```

Or for centralized mode:

```
~/.cce/
├── repos/
│   ├── <sha1-of-repo-root>/       # per-repo isolation
│   │   ├── index.sqlite
│   │   ├── qdrant/
│   │   ├── agent.sqlite
│   │   └── index.json
│   └── <sha1-of-another-repo>/
└── config.toml                    # global defaults
```

### Key Changes Required

---

### Component 1: Config — Repo-Aware Settings

#### [MODIFY] [config.py](file:///d:/Learn/code-context-extractor/cce/config.py)

1. Add `repo_root: Path | None = None` to top-level `Settings`
2. Make `PathsSettings` resolve relative to `repo_root`, not CWD:
   ```python
   def resolve(self, repo_root: Path) -> "PathsSettings":
       """Return a copy with all paths anchored to repo_root."""
       root = repo_root.resolve()
       return PathsSettings(
           data_dir=root / self.data_dir,
           sqlite_db=root / self.sqlite_db,
           ...
       )
   ```
3. Replace `get_settings()` singleton with a repo-aware factory:
   ```python
   def get_settings(repo_root: Path | None = None) -> Settings:
       settings = Settings()
       if repo_root:
           settings.paths = settings.paths.resolve(repo_root)
           settings.repo_root = repo_root
       return settings
   ```

---

### Component 2: Retrieval — Remove Global Singletons

#### [MODIFY] [tools.py](file:///d:/Learn/code-context-extractor/cce/retrieval/tools.py)

1. Replace `@lru_cache(maxsize=1)` on `_pipeline()` and `_hybrid_retriever()` with a **repo-keyed cache**:
   ```python
   _pipelines: dict[Path, IndexPipeline] = {}

   def _pipeline(repo_root: Path | None = None) -> IndexPipeline:
       root = repo_root or get_settings().repo_root or Path.cwd()
       if root not in _pipelines:
           settings = get_settings(repo_root=root)
           _pipelines[root] = IndexPipeline(settings=settings)
       return _pipelines[root]
   ```
2. Thread `repo_root` through all tool functions as an optional param

---

### Component 3: Vector Store — Consistent Collection Naming

#### [MODIFY] [vector_store.py](file:///d:/Learn/code-context-extractor/cce/index/vector_store.py)

1. Remove `collection_name_from_db()` — it's always wrong
2. Use only `collection_name(root)` everywhere
3. Store `repo_root` in the manifest and read it at query time

---

### Component 4: CLI — Explicit Root Passing

#### [MODIFY] [cli.py](file:///d:/Learn/code-context-extractor/cce/cli.py)

1. Add a global `--repo` option that sets the root for all subcommands:
   ```
   cce --repo /path/to/myapp query "where is login?"
   cce --repo /path/to/myapp search "auth" --mode hybrid
   ```
2. If `--repo` is not specified, auto-detect by walking up from CWD looking for `.cce/` (git-style)
3. Store the resolved root in the config so downstream code can find it

---

### Component 5: Server — Multi-Repo API

#### [MODIFY] [agent.py (server route)](file:///d:/Learn/code-context-extractor/cce/server/routes/agent.py)

1. Add `repo_root: str` to `AgentQuery`:
   ```python
   class AgentQuery(BaseModel):
       query: str
       thread_id: str = "default"
       repo_root: str | None = None  # path to the indexed repo
   ```
2. Resolve the pipeline from the repo root, not from the global singleton

---

### Component 6: Agent — Repo-Aware System Prompt

#### [MODIFY] [llm.py](file:///d:/Learn/code-context-extractor/cce/agents/llm.py)

1. Inject repo metadata into the system prompt:
   ```python
   SYSTEM_PROMPT = """\
   You are Code Context Engine — an expert code analysis assistant.
   You are querying the codebase at: {repo_root}
   Detected frameworks: {frameworks}
   Total indexed files: {file_count}
   Languages: {languages}
   ...
   """
   ```
2. Read this from the index manifest (`.cce/index.json`) at query time

---

### Component 7: Eval — Dynamic Dataset Support

#### [MODIFY] [core.yaml](file:///d:/Learn/code-context-extractor/cce/eval/datasets/core.yaml) + [harness.py](file:///d:/Learn/code-context-extractor/cce/eval/harness.py)

1. Add a `self-test` dataset that's valid only for CCE's own repo
2. Add a `generic` eval mode that tests retrieval quality without repo-specific expected symbols:
   - Does `search_code("auth")` return results?
   - Does `list_routes()` return at least 1 route?
   - Does `list_files()` return file paths?

---

### Component 8: Walker — Configurable Skip List

#### [MODIFY] [walker.py](file:///d:/Learn/code-context-extractor/cce/walker.py)

1. Split `_ALWAYS_SKIP` into two sets:
   - `_HARD_SKIP` = `{".git", ".venv", "node_modules", "__pycache__", ".mypy_cache"}` — never useful
   - `_SOFT_SKIP` = `{"migrations", "vendor", "target", "dist", "build", ".next"}` — skipped by default but overridable
2. Merge `.gitignore` patterns into the skip set (use `pathspec` library) so the walker respects what git ignores
3. Expose `--include migrations` / `--skip migrations` CLI flags that toggle entries in `_SOFT_SKIP`
4. Record the effective skip list in the manifest for reproducibility

---

### Component 9: Framework Detection — AST-Based

#### [MODIFY] [framework_detector.py](file:///d:/Learn/code-context-extractor/cce/extractors/framework_detector.py)

1. Accept a pre-walked `list[Path]` from `walk_repo()` instead of calling `rglob` — avoids re-walking `node_modules`
2. Replace substring matching with AST-level checks:
   ```python
   def _has_fastapi_import(path: Path) -> bool:
       tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
       for node in ast.walk(tree):
           if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fastapi"):
               return True
       return False
   ```
3. Cache the per-file decision in the manifest keyed by `(path, content_hash)` so re-detection is O(changed files)

---

### Component 10: FastAPI / Django Extractor Completeness

#### [MODIFY] [fastapi_extractor.py](file:///d:/Learn/code-context-extractor/cce/extractors/fastapi_extractor.py)

Add handling for:
- `app.add_api_route("/path", handler, methods=[...])` — treat as equivalent to a decorated route
- `@app.websocket("/ws")` — emit Route with `method="WEBSOCKET"`
- `app.mount("/prefix", sub_app)` — recursively extract from `sub_app`, prepend prefix to all its routes
- `APIRouter(prefix="/v1")` inline prefix resolution (check the constructor call, not just `include_router`)

#### [MODIFY] [django_extractor.py](file:///d:/Learn/code-context-extractor/cce/extractors/django_extractor.py)

Add handling for:
- `include('myapp.urls')` → recursively resolve and prepend the include prefix (mirror the F4 FastAPI fix)
- DRF `router.register(r'users', UserViewSet)` → emit synthetic routes for `GET/POST /users/`, `GET/PUT/PATCH/DELETE /users/{pk}/`, and one per `@action(...)` method on the ViewSet
- `@action(detail=True/False, methods=[...])` decorator on ViewSet methods
- `admin.site.register(Model, ModelAdmin)` → optional, emits admin routes

---

### Component 11: Cross-Stack Glue — `CALLS_API` Edges

#### [NEW] `cce/extractors/api_linker.py`

Runs as a second pass after all per-file extractors. Logic:
1. Query all `Route` / `URLPattern` nodes → build a trie keyed by `effective_path`.
2. Walk all JSX/TSX files for string literals passed to `fetch(...)`, `axios.get/post/...`, `useQuery({ url: ... })`, etc.
3. For each match, emit `CALLS_API` edge from the enclosing component symbol → the Route node.
4. Same pass for `.py` files calling `requests.get("/internal/...")` (useful for microservice graphs).

Enables:
- `find_callers(route_qname)` returns React components that hit that endpoint.
- `get_neighborhood(component, edge_kinds=["CALLS_API"])` traces request flow end-to-end.

---

### Component 12: JS/TS Reference Resolver

#### [MODIFY] [js_resolver.py](file:///d:/Learn/code-context-extractor/cce/parsers/js_resolver.py)

Phase-1 cheapest path (no `tsserver`):
1. Build an import table per file from tree-sitter `import_statement` / `import_clause` nodes.
2. For each `call_expression` whose callee is an Identifier, look up the binding in the import table → resolve to the exporting file/symbol.
3. Emit `CALLS` / `REFERENCES` edges the same way Python does.

Phase-2 (later): spawn a long-running `tsserver` subprocess for full type-aware resolution.

---

### Component 13: Node ID Salting for Repo Isolation

#### [MODIFY] [schema.py](file:///d:/Learn/code-context-extractor/cce/graph/schema.py)

```python
def _node_id_from_qname(qname: str, repo_salt: str = "") -> str:
    return hashlib.sha1(f"{repo_salt}::{qname}".encode()).hexdigest()[:16]
```

Pass `repo_salt = short_sha1(repo_root)` from the indexer. Makes centralized-mode safe and prevents accidental cross-repo collisions in shared DBs.

---

## Decisions (previously Open Questions)

- **Q1: Per-repo `.cce/` vs centralized.** → **Per-repo `.cce/` as default**, git-ignored, matches `.git/` ergonomics. Centralized (`~/.cce/repos/<sha1>/`) is an opt-in flag (`--central` or `CCE_STORAGE=central`). The layout is identical in both modes so code paths don't fork.
- **Q2: Qdrant collection naming.** → Always derive from `repo_root` (never from `db_path`). Delete `collection_name_from_db()`. Every consumer (indexer, retriever, doctor) reads `repo_root` from either the `Settings.repo_root` field or the manifest at startup — single source of truth.
- **Q3: Repo auto-detection.** → **Walk-up detection** (option B): walk up from CWD looking for `.cce/index.json`. If none found, fall back to CWD with a warning. `--repo` flag always overrides. Matches git's UX exactly.
- **Q4: Scope.** → Ship in phases. Phase 1 (items #1, #2, #3, #6) unblocks everything else and is ~1 day of work. Phase 2 (#4, #5, #7, #9) is incremental. Phase 3 (framework completeness) is its own project.

---

## Phased Execution Order

### **Phase 1 — Foundation (blocks everything else, ~1 day)**
1. **F-M1** `config.py`: add `repo_root`, `PathsSettings.resolve(root)`, repo-aware `get_settings()`.
2. **F-M2** `retrieval/tools.py` + `db.py`: replace `lru_cache(maxsize=1)` with `dict[Path, ...]` keyed caches.
3. **F-M3** `vector_store.py`: delete `collection_name_from_db()`, standardize on `collection_name(root)`. Fixes the silent vector-search bug.
4. **F-M4** `cli.py`: add `--repo` global option with walk-up auto-detect.
5. **F-M5** Regression test: index two different repos from a third CWD, query each, verify isolation.

### **Phase 2 — Correctness & UX (~2 days)**
6. **F-M6** `walker.py`: split hard/soft skip, add `.gitignore` pass, expose CLI flags. Fixes Django migration blind spot.
7. **F-M7** `server/routes/agent.py` + `index.py`: add `repo_root` field, resolve pipeline per request.
8. **F-M8** `agents/llm.py`: inject repo metadata into system prompt from the manifest.
9. **F-M9** `eval/harness.py`: add `generic` mode + `self-test` dataset tag.
10. **F-M10** `framework_detector.py`: AST-based, reuses walked file list, caches by content hash.

### **Phase 3 — Framework Completeness (~1 week, parallelizable)**
11. **F-M11** FastAPI extractor: `add_api_route`, websocket, mount, inline prefix.
12. **F-M12** Django extractor: `include()` resolution, DRF `router.register`, `@action`.
13. **F-M13** `api_linker.py` (new): emit `CALLS_API` edges between React/Python callers and routes.
14. **F-M14** `js_resolver.py`: import-table-based reference resolution (tree-sitter only, no tsserver).
15. **F-M15** Node-id salting for repo isolation.

---

## Verification Plan

### Automated Tests
1. Index CCE's own repo → `cce query "where is planner_node?"` → must return correct answer
2. Index a **second** repo (small Django project) from a **different CWD** → queries must hit the correct DB
3. Run `cce doctor` for both repos → both show valid manifests with correct `repo_root`
4. `cce eval-agent` against both repos → no crashes, reasonable scores
5. Vector search round-trip test: index → immediately search → assert at least 1 hit (catches collection-name regressions)

### Manual Verification
1. Index repo A from directory X → Index repo B from directory Y → Query repo A from directory Z → verify correct results
2. Check that `.cce/` is created inside the target repo, not the CWD
3. Index a React+FastAPI monorepo → `find_callers` on a FastAPI route returns the React components (post-Phase-3)
4. Index a Django project with `include('app.urls')` → `list_routes()` returns the prefixed paths (post-Phase-3)
