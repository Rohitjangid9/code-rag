"""SCIP-aligned Node / Edge dataclasses shared across all layers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


class NodeKind(str, Enum):
    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    VARIABLE = "Variable"
    ROUTE = "Route"
    MODEL = "Model"
    COMPONENT = "Component"
    HOOK = "Hook"
    SERIALIZER = "Serializer"
    PYDANTIC_MODEL = "PydanticModel"
    MIDDLEWARE = "Middleware"
    SIGNAL = "Signal"
    URL_PATTERN = "URLPattern"
    CLI_COMMAND = "CliCommand"


class EdgeKind(str, Enum):
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    INHERITS = "INHERITS"
    DECORATES = "DECORATES"
    REFERENCES = "REFERENCES"
    RETURNS_TYPE = "RETURNS_TYPE"
    PARAM_TYPE = "PARAM_TYPE"
    RAISES = "RAISES"
    USES_MODEL = "USES_MODEL"
    ROUTES_TO = "ROUTES_TO"
    RENDERS = "RENDERS"
    USES_HOOK = "USES_HOOK"
    USES_PROP = "USES_PROP"
    HANDLES_SIGNAL = "HANDLES_SIGNAL"
    DEPENDS_ON = "DEPENDS_ON"
    MOUNTS_ROUTER = "MOUNTS_ROUTER"
    CALLS_API = "CALLS_API"


class Language(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    TSX = "tsx"
    JSX = "jsx"
    GO = "go"       # F29
    JAVA = "java"   # F29
    RUST = "rust"   # F29


class FrameworkTag(str, Enum):
    DJANGO = "django"
    DRF = "drf"
    FASTAPI = "fastapi"
    REACT = "react"


class Location(BaseModel):
    file: str
    line: int
    col: int = 0


class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(ULID()))
    kind: NodeKind
    qualified_name: str
    name: str
    file_path: str
    line_start: int
    line_end: int
    signature: str | None = None
    docstring: str | None = None
    language: Language
    framework_tag: FrameworkTag | None = None
    visibility: str = "public"
    content_hash: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    src_id: str
    dst_id: str
    kind: EdgeKind
    location: Location | None = None
    confidence: float = 1.0


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(ULID()))
    node_id: str
    header: str
    body: str
    token_count: int = 0


class SubGraph(BaseModel):
    root_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
