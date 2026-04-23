"""F-M13 — cross-stack API linker.

Post-indexing pass that converts heuristic ``REFERENCES`` edges produced by the
JS/TS resolver (with ``dst_qualified_name`` starting with ``"api:"``) into
precise ``CALLS_API`` edges that point at the concrete backend ``Route`` /
``URLPattern`` symbol.

The matcher is path-based with a last-resort prefix fallback: the frontend
string ``/api/users/42`` matches a backend route ``/api/users/{id}`` because
the templated segments collapse to wildcards.  When several routes match, the
one with the longest literal prefix wins — this approximates the Django /
FastAPI ordering semantics without a full router simulation.
"""

from __future__ import annotations

import json
import re
import sqlite3

from cce.graph.schema import EdgeKind
from cce.logging import get_logger

log = get_logger(__name__)

_ROUTE_KINDS = ("Route", "URLPattern")
_PARAM_RE = re.compile(r"\{[^/]+\}|<[^/]+>|:[A-Za-z_][A-Za-z0-9_]*")


def _normalise_path(path: str) -> str:
    """Collapse templated segments (``{id}``/``<int:pk>``/``:id``) to ``*``."""
    if not path:
        return ""
    # Strip trailing slash for consistent comparison, except for the root.
    stripped = path.rstrip("/") or "/"
    return _PARAM_RE.sub("*", stripped)


def _route_paths(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Return ``(route_id, effective_path, normalised_path)`` for every route.

    ``effective_path`` falls back to ``meta.path`` (FastAPI) or the symbol name
    (Django URLPattern) when no cross-file prefix was stamped.
    """
    placeholders = ",".join("?" * len(_ROUTE_KINDS))
    rows = conn.execute(
        f"SELECT id, name, meta FROM symbols WHERE kind IN ({placeholders})",
        _ROUTE_KINDS,
    ).fetchall()
    out: list[tuple[str, str, str]] = []
    for row in rows:
        meta = json.loads(row["meta"] or "{}")
        path = meta.get("effective_path") or meta.get("path") or row["name"] or ""
        if not path:
            continue
        out.append((row["id"], path, _normalise_path(path)))
    return out


def _segments(path: str) -> list[str]:
    return [s for s in path.split("/") if s]


def _match_score(call_path: str, route_norm: str) -> int:
    """Return a score for how well a frontend call matches a route, or ``-1``.

    Scoring favours:
    * exact matches (max score),
    * equal segment count with literal segments matching,
    * the longest shared literal prefix as a tiebreaker.
    """
    call_norm = _normalise_path(call_path)
    if call_norm == route_norm:
        return 10_000
    call_segs = _segments(call_norm)
    route_segs = _segments(route_norm)
    if len(call_segs) != len(route_segs):
        return -1
    literal_matches = 0
    for cs, rs in zip(call_segs, route_segs):
        if rs == "*":
            continue
        if cs != rs:
            return -1
        literal_matches += 1
    return literal_matches


def link_api_references(conn: sqlite3.Connection) -> int:
    """Emit ``CALLS_API`` edges for every resolvable ``api:...`` REFERENCES edge.

    Returns the number of edges created.  Existing ``CALLS_API`` rows with the
    same ``(src_id, dst_id, file_path, line)`` key are skipped so the pass is
    idempotent on re-index.
    """
    routes = _route_paths(conn)
    if not routes:
        return 0

    refs = conn.execute(
        "SELECT src_id, path, method, file_path, line, confidence FROM api_refs",
    ).fetchall()

    created = 0
    for ref in refs:
        call_path = (ref["path"] or "").strip()
        if not call_path:
            continue
        best_score = -1
        best_route_id: str | None = None
        for route_id, _route_path, route_norm in routes:
            score = _match_score(call_path, route_norm)
            if score > best_score:
                best_score = score
                best_route_id = route_id
        if best_route_id is None or best_score < 0:
            continue
        # Skip if a CALLS_API edge for this same call site already exists.
        exists = conn.execute(
            "SELECT 1 FROM edges WHERE src_id=? AND dst_id=? AND kind=? "
            "AND file_path=? AND line=?",
            (ref["src_id"], best_route_id, EdgeKind.CALLS_API.value,
             ref["file_path"] or "", ref["line"] or 0),
        ).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, file_path, line, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                ref["src_id"], best_route_id, EdgeKind.CALLS_API.value,
                ref["file_path"] or "", ref["line"] or 0,
                float(ref["confidence"] or 0.7),
            ),
        )
        created += 1
    conn.commit()
    if created:
        log.info("api_linker: created %d CALLS_API edges", created)
    return created
