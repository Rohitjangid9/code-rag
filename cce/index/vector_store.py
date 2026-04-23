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
        """Deterministic collection name from the repo root path.

        F-M3: single source of truth — every producer and consumer hashes the
        repo root.  The previous ``collection_name_from_db`` helper was removed
        because it hashed a different path and silently broke vector search.
        """
        h = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
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
            return

        # Collection exists — check vector dimension matches current embedder
        info = self._client.get_collection(collection)
        current_size = info.config.params.vectors.size  # type: ignore[union-attr]
        if current_size != self._dim:
            log.warning(
                "Collection %r dim mismatch (%d vs %d) — recreating",
                collection, current_size, self._dim,
            )
            self._client.delete_collection(collection)
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            log.info("Recreated Qdrant collection %r (dim=%d)", collection, self._dim)

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
                    "content_hash": c.content_hash,  # F21
                },
            )
            for c, vec in chunks_and_vectors
        ]
        self._client.upsert(collection_name=collection, points=points)

    def get_existing_hashes(self, collection: str, node_ids: list[str]) -> dict[str, str]:
        """Return {node_id: content_hash} for all existing vectors for *node_ids* (F21).

        Used to skip re-embedding chunks whose content hasn't changed.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny  # noqa: PLC0415

        if not node_ids or not self.collection_exists(collection):
            return {}
        try:
            resp = self._client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="node_id", match=MatchAny(any=node_ids))]
                ),
                with_payload=True,
                limit=len(node_ids) * 3,  # one node may have multiple chunks
            )
            result: dict[str, str] = {}
            for point in resp[0]:
                nid = (point.payload or {}).get("node_id", "")
                h = (point.payload or {}).get("content_hash", "")
                if nid and h:
                    result[nid] = h
            return result
        except Exception:  # noqa: BLE001
            return {}

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

        resp = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return resp.points


def _chunk_uuid(chunk_id: str) -> str:
    """Convert ULID string to a UUID-like hex for Qdrant point IDs."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
