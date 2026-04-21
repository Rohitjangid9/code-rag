"""Phase 11 — SCIP (Semantic Code Intelligence Protocol) data types.

Follows the SCIP spec (https://github.com/sourcegraph/scip) in JSON form.
Protobuf binary emission is deferred; this module outputs the textproto-compatible
JSON representation that can be converted with scip convert.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum


class SymbolRole(IntEnum):
    UNSPECIFIED_SYMBOL_ROLE = 0
    DEFINITION = 1
    IMPORT = 2
    WRITE_ACCESS = 4
    READ_ACCESS = 8
    GENERATED = 16
    TEST = 32
    FORWARD_DEFINITION = 64


class RelationshipKind(str):
    """SCIP relationship types as string constants."""
    IS_REFERENCE = "is_reference"
    IS_IMPLEMENTATION = "is_implementation"
    IS_TYPE_DEFINITION = "is_type_definition"
    OVERRIDES = "overrides"


@dataclass
class SCIPPosition:
    """0-based [start_line, start_char, end_line, end_char]."""
    start_line: int
    start_char: int = 0
    end_line: int = -1
    end_char: int = -1

    def as_list(self) -> list[int]:
        end_line = self.end_line if self.end_line >= 0 else self.start_line
        return [self.start_line, self.start_char, end_line, self.end_char if self.end_char >= 0 else 0]


@dataclass
class SCIPRelationship:
    symbol: str
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    overrides: bool = False

    def as_dict(self) -> dict:
        d: dict = {"symbol": self.symbol}
        if self.is_reference:
            d["is_reference"] = True
        if self.is_implementation:
            d["is_implementation"] = True
        if self.is_type_definition:
            d["is_type_definition"] = True
        if self.overrides:
            d["overrides"] = True
        return d


@dataclass
class SCIPSymbolInfo:
    symbol: str
    documentation: list[str] = field(default_factory=list)
    relationships: list[SCIPRelationship] = field(default_factory=list)

    def as_dict(self) -> dict:
        d: dict = {"symbol": self.symbol}
        if self.documentation:
            d["documentation"] = self.documentation
        if self.relationships:
            d["relationships"] = [r.as_dict() for r in self.relationships]
        return d


@dataclass
class SCIPOccurrence:
    range: list[int]               # [start_line, start_char, end_line, end_char]
    symbol: str
    symbol_roles: int = 0
    override_documentation: list[str] = field(default_factory=list)
    diagnostics: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d: dict = {"range": self.range, "symbol": self.symbol}
        if self.symbol_roles:
            d["symbol_roles"] = self.symbol_roles
        return d


@dataclass
class SCIPDocument:
    relative_path: str
    language: str
    occurrences: list[SCIPOccurrence] = field(default_factory=list)
    symbols: list[SCIPSymbolInfo] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "language": self.language,
            "occurrences": [o.as_dict() for o in self.occurrences],
            "symbols": [s.as_dict() for s in self.symbols],
        }


@dataclass
class SCIPToolInfo:
    name: str = "code-context-engine"
    version: str = "0.1.0"
    arguments: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "arguments": self.arguments}


@dataclass
class SCIPMetadata:
    version: int = 0
    tool_info: SCIPToolInfo = field(default_factory=SCIPToolInfo)
    project_root: str = ""
    text_document_encoding: int = 1  # UTF8

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "tool_info": self.tool_info.as_dict(),
            "project_root": self.project_root,
            "text_document_encoding": self.text_document_encoding,
        }


@dataclass
class SCIPIndex:
    metadata: SCIPMetadata = field(default_factory=SCIPMetadata)
    documents: list[SCIPDocument] = field(default_factory=list)
    external_symbols: list[SCIPSymbolInfo] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "metadata": self.metadata.as_dict(),
            "documents": [d.as_dict() for d in self.documents],
            "external_symbols": [s.as_dict() for s in self.external_symbols],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)
