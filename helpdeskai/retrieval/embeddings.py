"""Embedding model adapter used by retrieval indexing and search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """Thin adapter over SentenceTransformers with model-specific prefixes."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            return int(self._model.get_sentence_embedding_dimension())
        return int(self._model.get_embedding_dimension())

    def _prefix(self, text: str, kind: str) -> str:
        if "e5" in self.model_name.lower():
            return f"{kind}: {text}"
        return text

    def encode_documents(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Embed corpus chunks."""
        return np.asarray(
            self._model.encode(
                [self._prefix(text, "passage") for text in texts],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Embed one user query."""
        return np.asarray(
            self._model.encode(
                self._prefix(query, "query"),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
