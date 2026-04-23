"""Shared output types for all parsers: ParsedFile, RawEdge, Parser protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cce.graph.schema import EdgeKind, Language, Node


@dataclass
class RawEdge:
    """An edge before dst is resolved to a Node id (might only have a name)."""

    src_id: str
    dst_qualified_name: str        # resolve → dst_id during graph write
    kind: EdgeKind
    file_path: str = ""
    line: int = 0
    col: int = 0
    confidence: float = 1.0
    # How this edge was produced — essential for debugging bad graph edges.
    # Values: "tree-sitter" | "jedi" | "ts-morph" | "heuristic" | "name-match" | "import"
    resolver_method: str = ""
    # Optional metadata to MERGE into the destination symbol when the edge
    # resolves successfully (e.g. {"graph_node_name": "planner"} from LangGraph).
    # Does NOT override existing meta keys — uses a JSON merge-patch approach.
    dst_meta_patch: dict = field(default_factory=dict)


@dataclass
class ParsedFile:
    """Output of one parser pass over a single source file."""

    path: Path
    rel_path: str                  # repo-relative string path
    language: Language
    nodes: list[Node] = field(default_factory=list)
    raw_edges: list[RawEdge] = field(default_factory=list)
    source: str = ""               # full source text (for lexical + Jedi)


class Parser(Protocol):
    """Protocol every parser must implement."""

    def parse(self, path: Path, rel_path: str, language: Language, source: str) -> ParsedFile:
        """Parse *source* and return nodes + raw edges."""
        ...
