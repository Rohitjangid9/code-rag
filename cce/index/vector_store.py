"""Phase 7 — Qdrant vector store wrapper (embedded or remote mode)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from cce.config import Settings
from cce.logging import get_logger

if TYPE_CHECKING:
    from cce.embeddings.chunker import Chunk

log = get_logger(__name__)


class VectorStore:
    """Thin wrapper around Qdrant for code chunk storage and retrieval."""

    def __init__(self, settings: Settings) -> None:
        from qdrant_client import QdrantClient  # noqa: PLC0415
        from cce.embeddings.embedder import _resolve  # noqa: PLC0415

        cfg = settings.qdrant
        if cfg.mode == "embedded":
            self._client = QdrantClient(path=str(settings.paths.qdrant_path))
        else:
            self._client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
        # Resolve auto (dim=0) against the embedder's canonical table so the
        # Qdrant collection is created with the correct vector size.
        _, self._dim = _resolve(
            settings.embedder.backend, settings.embedder.model_name, settings.embedder.dim,
        )
        self._settings = settings

    # ── Collection management ─────────────────────────────────────────────────

    def collection_name(self, root: Path) -> str:
        """Deterministic collection name from the repo root path."""
        h = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
        return f"cce_{h}"

    def collection_name_from_db(self, db_path: Path) -> str:
        """Derive collection name from the SQLite DB path (used without root)."""
        h = hashlib.sha1(str(db_path.resolve()).encode()).hexdigest()[:12]
        return f"cce_{h}"

    def collection_exists(self, collection: str) -> bool:
        existing = {c.name for c in self._client.get_collections().collections}
        return collection in existing

    def ensure_collection(self, collection: str) -> None:
        from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

        if not self.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            log.info("Created Qdrant collection %r (dim=%d)", collection, self._dim)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, collection: str, chunks_and_vectors: list[tuple["Chunk", list[float]]]) -> None:
        from qdrant_client.models import PointStruct  # noqa: PLC0415

        points = [
            PointStruct(
                id=_chunk_uuid(c.chunk_id),
                vector=vec,
                payload={
                    "node_id": c.node_id,
                    "path": c.path,
                    "qualified_name": c.qualified_name,
                    "kind": c.kind,
                    "framework_tag": c.framework_tag,
                    "header": c.header,
                    "body_preview": c.body[:300],
                },
            )
            for c, vec in chunks_and_vectors
        ]
        self._client.upsert(collection_name=collection, points=points)

    def delete_for_node_ids(self, collection: str, node_ids: list[str]) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchAny  # noqa: PLC0415

        if not node_ids:
            return
        self._client.delete(
            collection_name=collection,
            points_selector=Filter(must=[FieldCondition(key="node_id", match=MatchAny(any=node_ids))]),
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, collection: str, query_vector: list[float],
               k: int = 10, filters: dict | None = None):
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        return self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )


def _chunk_uuid(chunk_id: str) -> str:
    """Convert ULID string to a UUID-like hex for Qdrant point IDs."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
