# CLI Reference

All commands below are exposed by the Typer app in `cce.cli` (entry point
`cce = "cce.cli:app"` in `pyproject.toml`). Run `cce --help` for the raw help.

## Index & inspect

### `cce scan <path>`
Walk a codebase and print a per-language file inventory (no index written).
Useful to validate the walker's `.gitignore` + language detection.

### `cce index <path> [--layers ...]`
Build index layers for the given codebase root.

| Flag | Default | Values |
|------|---------|--------|
| `--layers` | `lexical,symbols,graph,framework` | Any comma-joined subset of `lexical, symbols, graph, framework, semantic` |

Prints a summary: new / changed / deleted files, symbols, edges, elapsed
seconds, and up to 5 error messages. Adding `semantic` requires an embedder
backend configured in `.env` (see [`configuration.md`](./configuration.md)).

### `cce info`
Pretty-prints the fully resolved `Settings` object as JSON. Handy to confirm
which `.env` values were actually picked up.

### `cce doctor`
Runs dependency and configuration checks and prints a table with ✓ / ⚠ / ✗:
- `tree-sitter` grammars for `python`, `typescript`, `tsx`
- SQLite DB connection at `CCE_PATHS__SQLITE_DB`
- Embedded Qdrant at `CCE_PATHS__QDRANT_PATH`
- Embedder backend import + model dim resolution
- GPU presence (CUDA / MPS)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars
- Installed Python packages (`watchdog`, `ulid`, `yaml`, `langchain_core`,
  `langgraph`)
- Node.js (reserved for a future ts-morph sidecar)

Run this **first** after install — it's faster than debugging a failed index.

## Search & navigate

### `cce search <query> [--mode] [--k]`
Plain BM25 searches without the hybrid pipeline.

| Flag | Default | Values |
|------|---------|--------|
| `--mode` | `lexical` | `lexical` (source FTS5), `symbols` (symbol FTS5), `hybrid` (runs both) |
| `--k` | `10` | Top-k hits |

### `cce query <query> [--mode] [--k]`
Runs the full **hybrid retriever** (Phase 8): BM25 + vector + graph 1-hop
expansion fused with RRF. Returns a rich table with `score`, `provenance`,
`symbol/path`, `line`, `snippet`.

| Flag | Default | Values |
|------|---------|--------|
| `--mode` | `hybrid` | `hybrid`, `lexical` (symbol FTS only), `semantic` (Qdrant only — needs L4) |
| `--k` | `10` | |

### `cce symbols <path> [--kind]`
Lists every symbol stored for a file or directory path.
`--kind` filters by `NodeKind` (`Class`, `Function`, `Method`, `Component`,
`Route`, `Model`, …). The filter is case-insensitive.

### `cce get <qualified_name>`
Shows full JSON for one symbol (signature, docstring, file/line range,
framework tag, `meta`). Exits 1 if not found.

### `cce callers <qualified_name>`
Lists every symbol with an outgoing `CALLS` edge to the given symbol, printed
as `qualified_name (file:line)`.

### `cce refs <qualified_name>`
Lists every `REFERENCES`/typed-edge pointing to the symbol, printed as
`KIND src_id → dst_id (location)`.

### `cce neighborhood <qualified_name> [--depth]`
Prints the N-hop subgraph around the symbol: node count, edge count, every
node's `kind + qualified_name`, and every edge `src --KIND--> dst`.
Default depth is `2`.

## Live / incremental

### `cce watch <path> [--layers] [--debounce]`
Runs a `watchdog` observer and re-indexes any file that changes. Debounce
seconds (`--debounce`, default `1.0`) collapse rapid saves into a single pass.
Layer selection matches `cce index`. Stops cleanly on `Ctrl-C` (Unix and
Windows).

## Export & evaluate

### `cce export-scip <path> [--out]`
Writes the current symbol+edge index in Sourcegraph **SCIP JSON** format.
`--out` defaults to `index.scip.json`. The result is readable by the
`scip print` CLI and by anything that speaks SCIP.

### `cce eval <path> --queries <file.yaml> [--k]`
Runs the retrieval evaluation harness against a YAML gold set (see
`tests/fixtures/` for examples). Reports MRR@k, Recall@k, nDCG@k, and a
provenance breakdown as a Rich table. `--k` defaults to `10`.

## Framework navigation (Phase 6)

### `cce get-route <pattern_or_path>`
Resolve a URL pattern or concrete path to its handler + response model.
Matches both exact patterns (`/users/{user_id}`) and concrete paths (the
engine normalises `/api/v1/users/42` → `/users/{id}` before matching).
Prints a `RouteInfo` JSON object; exits `1` if no route matches.

### `cce get-api-flow <route_or_component>`
Returns the UI → API → handler → response-model chain for the given anchor
as a `CrossStackFlow` JSON object. Works on route patterns, concrete URLs,
or component names — the engine walks `ROUTES_TO` and resolves the response
model through the symbol store.

## Serve

### `cce serve [--host] [--port]`
Starts the FastAPI app via `uvicorn` using the factory
`cce.server.app:create_app`. Defaults come from
`CCE_SERVER__HOST` / `CCE_SERVER__PORT` (`127.0.0.1:8765`). The MCP endpoint
is always mounted at `POST /mcp` (no flag required).

- Swagger UI: `http://127.0.0.1:8765/docs`
- OpenAPI JSON: `http://127.0.0.1:8765/openapi.json`
- MCP JSON-RPC: `POST http://127.0.0.1:8765/mcp`
- MCP tool list: `GET http://127.0.0.1:8765/mcp/tools`

See [`api-reference.md`](./api-reference.md) for the endpoint surface.

## Exit codes

All commands return `0` on success and `1` when a lookup fails (`cce get`,
`cce callers`, `cce refs`, `cce neighborhood` on an unknown qname). Pipeline
errors are printed inline but do not fail the whole index (first 5 shown).

## Examples

```bash
# First-time setup on a Django + React mono-repo
cce doctor
cce index ./webapp --layers lexical,symbols,graph,framework
cce index ./webapp --layers semantic       # later, after configuring embedder

# Ask a question
cce query "where do we validate JWT tokens?"
cce query "User" --mode lexical

# Jump around
cce symbols ./webapp/apps/users/views.py
cce get apps.users.views.UserViewSet.retrieve
cce callers apps.users.models.User.save
cce neighborhood apps.users.models.User --depth 3

# Framework navigation
cce get-route "/api/v1/users/{user_id}"
cce get-api-flow "/api/v1/users/42"

# Keep the index fresh while you code
cce watch ./webapp

# Serve it
cce serve
```
