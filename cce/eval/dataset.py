"""Phase 12 — EvalDataset: loads YAML query files for the retrieval harness.

YAML format:
    queries:
      - id: auth_middleware
        query: "how is authentication middleware wired?"
        expected_symbols:
          - "app.middleware.AuthMiddleware"
          - "app.views.authenticate_user"
        expected_files:            # optional — for file-level recall
          - "app/middleware.py"
        notes: "optional human note"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EvalQuery:
    id: str
    query: str
    expected_symbols: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalDataset:
    queries: list[EvalQuery] = field(default_factory=list)
    # F-M9: identifies which repo this dataset is valid for.  ``"self"``
    # means CCE's own repo (core.yaml); ``None`` means repo-agnostic.
    target_repo: str | None = None
    name: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalDataset":
        """Load from a YAML file (see module docstring for format)."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        queries = []
        for q in raw.get("queries", []):
            queries.append(EvalQuery(
                id=str(q.get("id", "")),
                query=str(q.get("query", "")),
                expected_symbols=q.get("expected_symbols", []),
                expected_files=q.get("expected_files", []),
                notes=str(q.get("notes", "")),
            ))
        return cls(
            queries=queries,
            target_repo=raw.get("target_repo"),
            name=raw.get("name", path.stem),
        )

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)
