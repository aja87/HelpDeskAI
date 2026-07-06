"""Reranking adapters for advanced RAG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from helpdeskai.retrieval.models import SearchResult


class Reranker(Protocol):
    """Minimal reranker interface used by the RAG pipeline."""

    model_name: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Return candidates reordered by relevance."""


class CrossEncoderReranker:
    """BGE cross-encoder reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Score query/chunk pairs and return the best candidates."""
        if not candidates:
            return []
        pairs = [(query, candidate.content[:2_000]) for candidate in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        return [
            SearchResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                score=float(score),
                mode=candidate.mode,
                metadata=candidate.metadata,
                source_scores=candidate.source_scores | {"reranker": float(score)},
            )
            for candidate, score in ranked[:top_k]
        ]
