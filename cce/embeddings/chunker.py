"""Phase 7 — Semantic chunker: one chunk per symbol node with a rich header.

Header format (prepended to body so the embedding encodes full context):
    # path: app/users/views.py
    # symbol: app.users.views.UserViewSet.retrieve
    # kind: Method
    # language: python
    # framework: drf
    # docstring: Returns a single user by id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ulid import ULID

from cce.graph.schema import Node
from cce.logging import get_logger

log = get_logger(__name__)

# ── Token budgets (word-level approximation; 1 word ≈ 1.3 tokens) ─────────────
# Total budget ≈ 1 500 words  (~1 950 tokens) — fits in 2k context window.
# Header gets priority; body gets the remainder.
MAX_HEADER_WORDS: int = 100    # ~130 tokens — enough for 6 metadata lines + docstring
MAX_BODY_WORDS: int = 1_400    # ~1 820 tokens — main code body
MAX_TOTAL_WORDS: int = MAX_HEADER_WORDS + MAX_BODY_WORDS


@dataclass
class Chunk:
    """A single embeddable chunk tied to one symbol node."""
    chunk_id: str = field(default_factory=lambda: str(ULID()))
    node_id: str = ""
    path: str = ""
    qualified_name: str = ""
    kind: str = ""
    framework_tag: str | None = None
    header: str = ""
    body: str = ""
    header_word_count: int = 0
    body_word_count: int = 0

    @property
    def token_count(self) -> int:
        """Rough token estimate (1.3 words/token)."""
        return int((self.header_word_count + self.body_word_count) * 1.3)


def build_header(node: Node, max_words: int = MAX_HEADER_WORDS) -> str:
    """Build the metadata header block prepended before the code body.

    The docstring is clamped so the total header stays within *max_words*.
    """
    base_lines = [
        f"# path: {node.file_path}",
        f"# symbol: {node.qualified_name}",
        f"# kind: {node.kind.value}",
        f"# language: {node.language.value}",
    ]
    if node.framework_tag:
        base_lines.append(f"# framework: {node.framework_tag.value}")

    base_words = sum(len(l.split()) for l in base_lines)
    remaining = max_words - base_words

    if node.docstring and remaining > 5:
        doc = node.docstring.replace("\n", " ").strip()
        doc_words = doc.split()
        if len(doc_words) > remaining:
            doc = " ".join(doc_words[:remaining]) + " …"
        base_lines.append(f"# docstring: {doc}")

    return "\n".join(base_lines)


def _trim_body(body_lines: list[str], max_words: int) -> tuple[str, int, bool]:
    """Return (trimmed_body, word_count, was_truncated)."""
    words_seen = 0
    trimmed: list[str] = []
    for line in body_lines:
        trimmed.append(line)
        words_seen += len(line.split())
        if words_seen >= max_words:
            trimmed.append("# … (body truncated to fit token budget)")
            return "\n".join(trimmed), words_seen, True
    return "\n".join(trimmed), words_seen, False


def chunk_node(node: Node, source_lines: list[str],
               max_header_words: int = MAX_HEADER_WORDS,
               max_body_words: int = MAX_BODY_WORDS) -> Chunk:
    """Extract and chunk a single symbol node from its source lines."""
    start = max(0, node.line_start - 1)
    end = min(len(source_lines), node.line_end)
    body_lines = source_lines[start:end]

    body, body_wc, truncated = _trim_body(body_lines, max_body_words)
    if truncated:
        log.debug("Chunk body truncated for %s (%d words → %d)", node.qualified_name,
                  len(" ".join(body_lines).split()), max_body_words)

    header = build_header(node, max_words=max_header_words)
    header_wc = len(header.split())

    return Chunk(
        chunk_id=str(ULID()),
        node_id=node.id,
        path=node.file_path,
        qualified_name=node.qualified_name,
        kind=node.kind.value,
        framework_tag=node.framework_tag.value if node.framework_tag else None,
        header=header,
        body=body,
        header_word_count=header_wc,
        body_word_count=body_wc,
    )


def chunk_nodes(nodes: list[Node], file_lines: dict[str, list[str]]) -> list[Chunk]:
    """Chunk all nodes that have source lines available.

    Skips nodes whose file isn't in *file_lines* (e.g. built-ins).
    """
    chunks: list[Chunk] = []
    for node in nodes:
        lines = file_lines.get(node.file_path)
        if not lines:
            continue
        # Skip very small nodes (< 2 lines — probably fields or variables)
        if node.line_end - node.line_start < 1:
            continue
        chunks.append(chunk_node(node, lines))
    return chunks
