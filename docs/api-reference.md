# API Reference

`cce serve` boots a FastAPI app (factory: `cce.server.app:create_app`) that
mounts six REST routers plus an MCP JSON-RPC endpoint. All request/response
payloads are Pydantic models; full schemas live in Swagger at `/docs`.

Default base URL: `http://127.0.0.1:8765` (configurable via
`CCE_SERVER__HOST` / `CCE_SERVER__PORT`).

## 1 · Health

| Method | Path | Purpose |
|-------:|------|---------|
| `GET`  | `/health` | Liveness probe. Returns `{"status": "ok"}`. |

## 2 · Indexing — `/index/*`

| Method | Path | Body / Query | Returns |
|-------:|------|--------------|---------|
| `POST` | `/index` | `{ "path": str, "layers": [str, …] }` | `IndexStatus` |
| `GET`  | `/index/status` | — | `IndexStatus` |

`IndexRequest.layers` defaults to `["lexical","symbols","graph","semantic"]`.
The pipeline wiring behind these endpoints is active in later phases; the
CLI (`cce index`) is the supported surface today.

## 3 · Search — `/search`

| Method | Path | Body | Returns |
|-------:|------|------|---------|
| `POST` | `/search` | `{ "query": str, "mode": "auto"\|"lexical"\|"semantic"\|"hybrid", "k": int, "filters": dict? }` | `list[Hit]` |

A `Hit` has `node` (full `Node` or null), `path`, `line_start`, `line_end`,
`snippet`, `score`, and `provenance ∈ {lex, vec, graph, hybrid}`.

## 4 · Symbols — `/symbols/*`

| Method | Path | Query | Returns |
|-------:|------|-------|---------|
| `GET` | `/symbols/outline` | `path: str` | `list[Node]` |
| `GET` | `/symbols/callers` | `qname: str` | `list[Node]` |
| `GET` | `/symbols/refs` | `qname: str` | `list[Location]` |
| `GET` | `/symbols/implementations` | `qname: str` | `list[Node]` |
| `GET` | `/symbols/{qualified_name:path}` | — | `Node` (404 if absent) |

`{qualified_name:path}` accepts dots and slashes, so
`/symbols/app.users.views.UserViewSet` and
`/symbols/app/users/views.py::UserViewSet` both work.

## 5 · Graph — `/graph/*`

| Method | Path | Query / Body | Returns |
|-------:|------|--------------|---------|
| `GET` | `/graph/neighborhood` | `qname: str, depth: int=2, edge_kinds: list[str]?` | `SubGraph` |
| `GET` | `/graph/route` | `pattern_or_path: str` | `RouteInfo` |
| `GET` | `/graph/component` | `component_name: str` | `ComponentTree` |
| `GET` | `/graph/api-flow` | `route_or_component: str` | `CrossStackFlow` |

`SubGraph = { root_id, nodes: [Node], edges: [Edge] }`.
`RouteInfo = { pattern, methods, handler_qname, framework, request_model?, response_model? }`.
`ComponentTree = { component_qname, children: [str], hooks: [str], props: [str] }`.
`CrossStackFlow = { anchor, steps: [{kind, name, file?, line?, framework?, methods?}] }`.

## 6 · Agent — `/agent/*`

| Method | Path | Body | Returns |
|-------:|------|------|---------|
| `POST` | `/agent/query` | `{ "query": str, "thread_id": str? }` | SSE stream of LangGraph events (tokens, tool calls, final answer) |

The stream follows LangGraph's event envelope (`messages`,
`on_tool_start`, `on_tool_end`, `on_chain_end`). `thread_id` selects a
checkpointer thread so you can resume a conversation; checkpointer backend is
chosen by `CCE_AGENT__CHECKPOINTER` (`memory` | `sqlite`).

## 7 · MCP — `/mcp` (JSON-RPC 2.0)

`cce` speaks **Model Context Protocol (2024-11)** so Claude Desktop, Cursor,
Continue, and anything else MCP-aware can use its tools directly.

| Method | Path | Purpose |
|-------:|------|---------|
| `GET`  | `/mcp/tools` | Returns the raw list of MCP tool descriptors (`name`, `description`, `inputSchema`). Useful for debugging. |
| `POST` | `/mcp` | JSON-RPC 2.0 entry point. Supported methods: `initialize`, `tools/list`, `tools/call`. |

The exposed tool set (`cce.agents.tools.ALL_TOOLS`):

| Tool | What it does |
|------|--------------|
| `search_code` | Multi-mode code search (`lexical` / `semantic` / `hybrid` / `auto`). |
| `get_symbol` | Full record for a qualified symbol name. |
| `get_file_outline` | All symbols defined in a file. |
| `find_callers` | Callers of a symbol. |
| `find_references` | Source locations that reference a symbol. |
| `get_neighborhood` | N-hop subgraph around a symbol. |
| `get_route` | URL pattern → handler + response model (Django/FastAPI). |
| `get_component_tree` | React component → children + hooks + props. |
| `get_api_flow` | Anchor → Route → Handler → Model chain across frontend and backend. |

### `tools/call` example

```bash
curl -s http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "search_code",
      "arguments": { "query": "JWT validation", "mode": "hybrid", "k": 5 }
    }
  }'
```

### Register with Claude Desktop / Cursor / Continue

Point the client at the HTTP transport:

```json
{
  "mcpServers": {
    "cce": { "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

A future release will add a stdio transport (`cce serve --mcp`) for clients
that require subprocess MCP servers.

## 8 · CORS

CORS is enabled via `CCE_SERVER__CORS_ORIGINS` (defaults to `["*"]`), with
credentials, methods, and headers fully open. Tighten for production.

## 9 · Error shape

Every route raises `HTTPException` with a JSON body `{ "detail": str }`:

| Status | Meaning |
|-------:|---------|
| `404` | Symbol / route / component not found. |
| `422` | Pydantic validation error on query or body. |
| `501` | Endpoint reserved for a later build phase. |
| `500` | Unhandled error; check server logs. |

## 10 · Authentication

None in v1 — `cce serve` is intended for loopback use. Put it behind a
reverse proxy or add dependency-injected auth in
`cce.server.deps` if you expose it on a network.
