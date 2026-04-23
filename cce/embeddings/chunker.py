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

import hashlib
from dataclasses import dataclass, field

from ulid import ULID

from cce.graph.schema import Node
from cce.logging import get_logger

log = get_logger(__name__)

# ── Token budgets ─────────────────────────────────────────────────────────────
# F24: use tiktoken when available for token-accurate budgets; fall back to the
# word-level approximation (1 word ≈ 1.3 tokens) so the code works without it.
#
# Total budget ≈ 1 950 tokens — fits in a 2 k context window.
# Header gets priority (≈130 tokens); body gets the remainder (≈1 820 tokens).
MAX_HEADER_TOKENS: int = 130
MAX_BODY_TOKENS: int = 1_820
MAX_TOTAL_TOKENS: int = MAX_HEADER_TOKENS + MAX_BODY_TOKENS

# Legacy word-based constants (kept so callers that pass max_words= still work)
MAX_HEADER_WORDS: int = 100
MAX_BODY_WORDS: int = 1_400
MAX_TOTAL_WORDS: int = MAX_HEADER_WORDS + MAX_BODY_WORDS


# ── F24: tiktoken helpers ─────────────────────────────────────────────────────

_count_tokens_fn = None  # lazy-initialised on first call


def _count_tokens(text: str) -> int:
    """Count tokens in *text* using tiktoken when available (F24).

    Initialised lazily on the first call so ``tiktoken`` never downloads its
    encoding file at import time (which would block the test suite on cold
    environments with no network access).  Falls back to a word-count
    approximation (1.3 tokens/word) when tiktoken is not installed.
    """
    global _count_tokens_fn  # noqa: PLW0603
    if _count_tokens_fn is None:
        try:
            import tiktoken  # noqa: PLC0415
            enc = tiktoken.get_encoding("cl100k_base")
            _count_tokens_fn = lambda t: len(enc.encode(t, disallowed_special=()))  # noqa: E731
            log.debug("tiktoken loaded for token-accurate chunk budgets")
        except Exception:  # noqa: BLE001
            _count_tokens_fn = lambda t: int(len(t.split()) * 1.3)  # noqa: E731
            log.debug("tiktoken not available — using word-count approximation")
    return _count_tokens_fn(text)


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
    # F21: SHA-256 of (header + body); used to skip re-embed when unchanged
    content_hash: str = ""

    @property
    def token_count(self) -> int:
        """Rough token estimate (1.3 words/token)."""
        return int((self.header_word_count + self.body_word_count) * 1.3)


def _chunk_hash(header: str, body: str) -> str:
    """Return hex SHA-256 of the concatenated header + body (F21)."""
    content = (header + "\n" + body).encode("utf-8", errors="replace")
    return hashlib.sha256(content).hexdigest()


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


def _trim_body(
    body_lines: list[str],
    max_words: int = MAX_BODY_WORDS,
    max_tokens: int = MAX_BODY_TOKENS,
) -> tuple[str, int, bool]:
    """Return (trimmed_body, word_count, was_truncated).

    F24: uses tiktoken when available for exact token accounting; the word-count
    guard acts as a secondary backstop for the legacy ``max_words`` callers.
    """
    words_seen = 0
    tokens_seen = 0
    trimmed: list[str] = []
    for line in body_lines:
        trimmed.append(line)
        words_seen += len(line.split())
        tokens_seen += _count_tokens(line)
        if tokens_seen >= max_tokens or words_seen >= max_words:
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
        content_hash=_chunk_hash(header, body),  # F21
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
