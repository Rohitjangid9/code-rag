"""Pluggable embedder backends.

DEFAULT: OpenAI text-embedding-3-large (3072‑dim, requires OPENAI_API_KEY).
Switch via CCE_EMBED__BACKEND in .env — see .env.example for all options.

Backend   | Model                              | Dim  | Needs
--------- | ---------------------------------- | ---- | ---------------------
openai    | text-embedding-3-large             | 3072 | OPENAI_API_KEY
openai    | text-embedding-3-small             | 1536 | OPENAI_API_KEY
jina      | jinaai/jina-embeddings-v2-base-code | 768  | sentence-transformers
nomic     | nomic-ai/nomic-embed-code          | 3584 | ~7GB VRAM

F36: ``embed_query`` is LRU-cached per distinct text so repeated vector searches
for the same string skip the API/GPU call.

F37: when ``CCE_OFFLINE=true``, all network calls raise ``RuntimeError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


# Canonical dim for each well-known model (used for auto-detection)
MODEL_DIMS: dict[str, int] = {
    "jinaai/jina-embeddings-v2-base-code": 768,
    "jinaai/jina-embeddings-v2-base-en": 768,
    "jinaai/jina-embeddings-v3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-ai/nomic-embed-code": 3584,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}

_BACKEND_DEFAULTS: dict[str, tuple[str, int]] = {
    # backend → (default_model, default_dim)
    "jina":   ("jinaai/jina-embeddings-v2-base-code", 768),
    "openai": ("text-embedding-3-large", 3072),
    "nomic":  ("nomic-ai/nomic-embed-code", 3584),
}

# OpenAI v3 models that accept the 'dimensions' parameter
_OPENAI_V3 = {"text-embedding-3-small", "text-embedding-3-large"}


def _resolve(backend: str, model_name: str, dim: int) -> tuple[str, int]:
    """Return (resolved_model_name, resolved_dim), filling blanks from defaults."""
    default_model, default_dim = _BACKEND_DEFAULTS.get(backend, ("", 0))
    model = model_name or default_model
    if dim == 0:
        dim = MODEL_DIMS.get(model, default_dim)
    return model, dim


class Embedder(ABC):
    backend_name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of code/text chunks. Returns list[list[float]]."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


# ── Jina (default — CPU, no API key) ─────────────────────────────────────────

class JinaEmbedder(Embedder):
    """jinaai/jina-embeddings-v2-base-code via sentence-transformers.

    * 768-dim, Apache-2.0, runs on CPU.
    * 8k token context window (ideal for function/class chunks).
    * First call downloads ~550MB model to HuggingFace cache.
    """

    backend_name = "jina"

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        dim: int = 768,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        self._st = SentenceTransformer(model_name, trust_remote_code=True)
        self.dim = dim
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._st.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        # Jina v2 uses task prefix for asymmetric retrieval
        return self._st.encode(
            f"Represent this code query for searching relevant code: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


# ── OpenAI (API, best quality) ────────────────────────────────────────────────

class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-large (3072d) or text-embedding-3-small (1536d).

    Set CCE_EMBED__MODEL_NAME and OPENAI_API_KEY in .env.
    For custom truncated dims on v3 models, set CCE_EMBED__DIM (e.g. 1024).
    """

    backend_name = "openai"

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        dim: int = 3072,
        api_key: str | None = None,
    ) -> None:
        import os  # noqa: PLC0415

        # Ensure .env is loaded if OPENAI_API_KEY is missing
        if api_key is None and not os.getenv("OPENAI_API_KEY"):
            from dotenv import load_dotenv  # noqa: PLC0415
            load_dotenv(".env")

        from openai import OpenAI  # noqa: PLC0415
        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        self.dim = dim
        # Only pass 'dimensions' for v3 models AND when user has customised the dim
        native_dim = MODEL_DIMS.get(model_name, dim)
        self._custom_dim = dim if (model_name in _OPENAI_V3 and dim != native_dim) else None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict = {"input": list(texts), "model": self._model}
        if self._custom_dim:
            kwargs["dimensions"] = self._custom_dim
        resp = self._client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ── Nomic (local GPU, highest offline quality) ────────────────────────────────

class NomicEmbedder(Embedder):
    """nomic-ai/nomic-embed-code via HuggingFace transformers (~7GB VRAM, fp16)."""

    backend_name = "nomic"

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-code",
        dim: int = 3584,
        device: str = "auto",
        dtype: str = "fp16",
        max_tokens: int = 2048,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._tok = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                       "fp32": torch.float32}.get(dtype, torch.float16)
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch_dtype
        )
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self._model = self._model.to(device).eval()
        self.dim = dim
        self._device = device
        self._max_tokens = max_tokens
        self._torch = torch

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"search_query: {text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import torch.nn.functional as F  # noqa: PLC0415
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=self._max_tokens, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with self._torch.no_grad():
            out = self._model(**enc)
        token_emb = out[0]
        mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        pooled = self._torch.sum(token_emb * mask, 1) / self._torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(pooled, p=2, dim=1).cpu().tolist()


# ── F36: query-level LRU cache wrapper ───────────────────────────────────────

class _CachedEmbedder(Embedder):
    """Thin wrapper that caches ``embed_query`` calls via ``functools.lru_cache``.

    The cache is keyed on the query text and lives as long as the embedder
    instance.  ``embed_documents`` (bulk indexing) is intentionally *not*
    cached — batch calls are large and should not accumulate in memory.
    """

    def __init__(self, inner: Embedder, maxsize: int = 512) -> None:
        import functools  # noqa: PLC0415
        self._inner = inner
        self.dim = inner.dim
        self.backend_name = inner.backend_name

        @functools.lru_cache(maxsize=maxsize)
        def _cached_query(text: str) -> tuple:
            return tuple(inner.embed_query(text))

        self._cached_query = _cached_query

    def embed_documents(self, texts) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return list(self._cached_query(text))


# ── Factory ────────────────────────────────────────────────────────────────────

def get_embedder() -> Embedder:
    """Return the configured embedder (cached, offline-guarded).

    F36: wraps with _CachedEmbedder to avoid re-embedding repeated queries.
    F37: raises RuntimeError when ``CCE_OFFLINE=true``.
    """
    from cce.config import get_settings  # noqa: PLC0415
    settings = get_settings()

    # F37: offline guard
    if settings.offline:
        raise RuntimeError(
            "CCE is running in offline mode (CCE_OFFLINE=true). "
            "Embedding calls are disabled — use lexical search instead."
        )

    cfg = settings.embedder
    model, dim = _resolve(cfg.backend, cfg.model_name, cfg.dim)

    if cfg.backend == "jina":
        inner = JinaEmbedder(model_name=model, dim=dim, batch_size=cfg.batch_size)
    elif cfg.backend == "openai":
        inner = OpenAIEmbedder(model_name=model, dim=dim)
    elif cfg.backend == "nomic":
        inner = NomicEmbedder(model_name=model, dim=dim,
                              device=cfg.device, dtype=cfg.dtype, max_tokens=cfg.max_tokens)
    else:
        raise ValueError(
            f"Unknown embedder backend: {cfg.backend!r}. "
            "Set CCE_EMBED__BACKEND to jina | openai | nomic  (see .env.example)."
        )
    return _CachedEmbedder(inner)  # F36
