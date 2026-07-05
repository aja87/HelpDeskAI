"""Embedding clients used by the indexing workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from sentence_transformers import SentenceTransformer


class BGEEmbeddingClient:
    """Local embedding client backed by SentenceTransformers."""

    def __init__(
        self,
        *,
        model: str,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")

        self._normalize_embeddings = normalize_embeddings
        self._encoder: Any = SentenceTransformer(model, trust_remote_code=trust_remote_code)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of input texts."""

        if not texts:
            return []

        vectors = self._encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=True,
        )
        vectors_list = vectors.tolist()
        if len(vectors_list) != len(texts):
            raise RuntimeError(
                "BGE embeddings response size mismatch: "
                f"expected {len(texts)} vectors, received {len(vectors_list)}"
            )
        return vectors_list
