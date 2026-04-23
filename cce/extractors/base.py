"""Shared output types and protocol for all framework extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cce.graph.schema import Node
from cce.parsers.base import RawEdge


@dataclass
class ExtractedData:
    """Nodes + raw edges produced by a framework extractor for one file."""
    nodes: list[Node] = field(default_factory=list)
    raw_edges: list[RawEdge] = field(default_factory=list)
    router_prefixes: dict[str, str] = field(default_factory=dict)


class FrameworkExtractor(Protocol):
    """Protocol every framework extractor implements."""

    def can_handle(self, path: Path, source: str) -> bool:
        """Return True if this extractor should run on this file."""
        ...

    def extract(self, path: Path, rel_path: str, source: str) -> ExtractedData:
        """Extract framework-specific nodes and edges from *source*."""
        ...
