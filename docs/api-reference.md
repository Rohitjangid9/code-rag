# API Reference

`cce serve` boots a FastAPI app (factory: `cce.server.app:create_app`) that
mounts six REST routers plus an MCP JSON-RPC endpoint. Every response model
is a Pydantic class — full, typed schemas live in Swagger at `/docs`.

Default base URL: `http://127.0.0.1:8765` (change via
`CCE_SERVER__HOST` / `CCE_SERVER__PORT`).

## 1 · Health — `cce/server/routes/health.py`

| Method | Path | Purpose |
|-------:|------|---------|
| `GET`  | `/health` | Liveness probe. |

## 2 · Indexing — `/index/*` — `cce/server/routes/index.py`

| Method | Path | Body / Query | Returns |
|-------:|------|--------------|---------|
| `POST` | `/index` | `IndexRequest { path: str, layers: list[str] }` | `IndexStatus` |
| `GET`  | `/index/status` | — | `IndexStatus` |

`IndexRequest.layers` defaults to `["lexical","symbols","graph","semantic"]`.
Both endpoints currently return HTTP `501` — the CLI (`cce index`) is the
supported surface until the pipeline is wired through HTTP in a later phase.

## 3 · Search — `/search` — `cce/server/routes/search.py`

| Method | Path | Body | Returns |
|-------:|------|------|---------|
| `POST` | `/search` | `SearchRequest { query, mode, k, filters? }` | `SearchResponse { query, mode, hits: [Hit] }` |

`mode ∈ {"auto", "lexical", "semantic", "hybrid"}` (default `"auto"`).
A `Hit` = `{ node?: Node, path, line_start, line_end, snippet, score,
provenance: "lex" | "vec" | "graph" | "hybrid" }`.

## 4 · Symbols — `/symbols/*` — `cce/server/routes/symbols.py`

| Method | Path | Query | Returns |
|-------:|------|-------|---------|
| `GET` | `/symbols/outline` | `path: str` | `list[Node]` |
| `GET` | `/symbols/callers` | `qname: str` | `list[Node]` |
| `GET` | `/symbols/refs` | `qname: str` | `list[Location]` |
| `GET` | `/symbols/implementations` | `qname: str` | `list[Node]` |
| `GET` | `/symbols/{qualified_name:path}` | — | `Node` (404 if absent) |

The `:path` converter on the last route lets dots and slashes appear in the
qualified name (e.g. `/symbols/app.users.views.UserViewSet`).

## 5 · Graph — `/graph/*` — `cce/server/routes/graph.py`

| Method | Path | Query | Returns |
|-------:|------|-------|---------|
| `GET` | `/graph/neighborhood` | `qname: str, depth: int = 2` | `SubGraph` |
| `GET` | `/graph/route` | `pattern: str` | `RouteInfo` |
| `GET` | `/graph/component` | `name: str` | `ComponentTree` |
| `GET` | `/graph/api-flow` | `anchor: str` | `CrossStackFlow` |

Shapes (from `cce.retrieval.tools`):

```
SubGraph       = { root_id, nodes: [Node], edges: [Edge] }
RouteInfo      = { pattern, methods: [str], handler_qname, framework,
                   request_model?, response_model? }
ComponentTree  = { component_qname, children: [str], hooks: [str], props: [str] }
CrossStackFlow = { anchor, steps: [ {kind, name, file?, line?,
                                     framework?, methods?} ] }
```

## 6 · Agent — `/agent/*` — `cce/server/routes/agent.py`

| Method | Path | Body | Returns |
|-------:|------|------|---------|
| `POST` | `/agent/query`  | `AgentQuery { query, thread_id?: str }` | `AgentResponse { answer, citations: [dict], reasoning_steps: [str] }` — synchronous, single JSON reply. |
| `POST` | `/agent/stream` | `AgentQuery` | **Server-Sent Events** stream of LangGraph `astream_events(version="v2")` output. |

`thread_id` (default `"default"`) selects the LangGraph checkpointer thread
so conversations resume. The checkpointer backend is set by
`CCE_AGENT__CHECKPOINTER` (`sqlite` or `memory`).

## 7 · MCP — `/mcp`, `/mcp/tools` — `cce/server/mcp.py`

`cce` speaks **Model Context Protocol (2024-11-05)** so Claude Desktop,
Cursor, Continue, and anything else MCP-aware can use its tools directly.

| Method | Path | Purpose |
|-------:|------|---------|
| `GET`  | `/mcp/tools` | Returns `{tools: [{name, description, inputSchema}, …]}`. Convenience endpoint (not part of the spec). |
| `POST` | `/mcp` | JSON-RPC 2.0 entry point. Supported methods: `initialize`, `tools/list`, `tools/call`. Unknown methods return RPC error `-32601`. |

The exposed tool set is `cce.agents.tools.ALL_TOOLS`:

| Tool | What it does |
|------|--------------|
| `search_code` | Multi-mode code search (`lexical` / `semantic` / `hybrid` / `auto`). |
| `get_symbol` | Full record for a qualified symbol name. |
| `get_file_outline` | All symbols defined in a file. |
| `find_callers` | Callers of a symbol. |
| `find_references` | Source locations referencing a symbol. |
| `get_neighborhood` | N-hop subgraph around a symbol. |
| `get_route` | URL pattern → handler + response model (Django/FastAPI). |
| `get_component_tree` | React component → children + hooks + props. |
| `get_api_flow` | Anchor → Route → Handler → Model chain across stacks. |

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

The response `result.content[0].text` is a JSON-stringified version of the
tool return value. `isError: true` signals a tool failure.

### Registering with MCP clients

```json
{
  "mcpServers": {
    "cce": { "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

Only HTTP transport is supported today; stdio transport is not implemented.

## 8 · CORS

`CORSMiddleware` is added with `CCE_SERVER__CORS_ORIGINS` (default `["*"]`),
`allow_credentials=True`, and all methods / headers open. Tighten in
production deployments.

## 9 · Error shape

Routes raise `HTTPException` with body `{ "detail": str }`:

| Status | Meaning |
|-------:|---------|
| `404` | Symbol / route / component not found. |
| `422` | Pydantic validation error on query or body. |
| `501` | Endpoint reserved for a later build phase (today: the `/index/*` routes). |
| `500` | Unhandled error; check server logs. |

## 10 · Authentication

None in v1 — `cce serve` binds to loopback by default. Put it behind a
reverse proxy or wire dependency-injected auth via `cce/server/deps.py` if
you expose it on a network.
