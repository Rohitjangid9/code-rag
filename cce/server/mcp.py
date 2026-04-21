"""Phase 9 — MCP (Model Context Protocol) JSON-RPC 2.0 endpoint.

Exposes all CCE retrieval tools as MCP-compatible tool definitions.
Compatible with the MCP 2024-11 specification.

Endpoint: POST /mcp
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from cce.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["mcp"])


# ── JSON-RPC 2.0 models ───────────────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any = None
    error: dict | None = None


# ── Tool registry ─────────────────────────────────────────────────────────────

def _mcp_tool_list() -> list[dict]:
    """Convert ALL_TOOLS LangChain tools → MCP tool descriptors."""
    from cce.agents.tools import ALL_TOOLS  # noqa: PLC0415
    tools = []
    for t in ALL_TOOLS:
        schema = t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}}
        tools.append({
            "name": t.name,
            "description": t.description,
            "inputSchema": schema,
        })
    return tools


def _call_tool(name: str, arguments: dict) -> Any:
    from cce.agents.tools import ALL_TOOLS  # noqa: PLC0415
    tool_map = {t.name: t for t in ALL_TOOLS}
    tool = tool_map.get(name)
    if not tool:
        raise KeyError(f"Unknown tool: {name!r}")
    return tool.invoke(arguments)


# ── Dispatch table ────────────────────────────────────────────────────────────

def _handle_initialize(_params: dict | None) -> dict:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "code-context-engine", "version": "0.1.0"},
    }


def _handle_tools_list(_params: dict | None) -> dict:
    return {"tools": _mcp_tool_list()}


def _handle_tools_call(params: dict | None) -> dict:
    params = params or {}
    name = params.get("name", "")
    arguments = params.get("arguments", {})
    try:
        result = _call_tool(name, arguments)
        # MCP content format
        if isinstance(result, (list, dict)):
            import json  # noqa: PLC0415
            text = json.dumps(result, default=str, indent=2)
        else:
            text = str(result)
        return {"content": [{"type": "text", "text": text}], "isError": False}
    except KeyError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("Tool %r error: %s", name, exc)
        return {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}


_DISPATCH = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}


# ── FastAPI endpoint ──────────────────────────────────────────────────────────

@router.post("/mcp", response_model=JsonRpcResponse)
def mcp_endpoint(req: JsonRpcRequest) -> JsonRpcResponse:
    """JSON-RPC 2.0 MCP server. Handles initialize, tools/list, tools/call."""
    handler = _DISPATCH.get(req.method)
    if handler is None:
        return JsonRpcResponse(
            id=req.id,
            error={"code": -32601, "message": f"Method not found: {req.method}"},
        )
    try:
        result = handler(req.params)
        return JsonRpcResponse(id=req.id, result=result)
    except Exception as exc:  # noqa: BLE001
        log.error("MCP error for method %r: %s", req.method, exc)
        return JsonRpcResponse(
            id=req.id,
            error={"code": -32603, "message": "Internal error", "data": str(exc)},
        )


@router.get("/mcp/tools")
def mcp_tools_get() -> dict:
    """Convenience GET for tool discovery (not part of spec, useful for debugging)."""
    return {"tools": _mcp_tool_list()}
