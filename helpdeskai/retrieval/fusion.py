"""Hybrid retrieval fusion helpers."""

from __future__ import annotations

from collections.abc import Sequence

from helpdeskai.retrieval.models import SearchMode, SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fuse rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}
    source_scores: dict[str, dict[str, float]] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(result.chunk_id, result)
            source_scores.setdefault(result.chunk_id, {}).update(result.source_scores)

    fused_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    return [
        SearchResult(
            chunk_id=by_id[chunk_id].chunk_id,
            document_id=by_id[chunk_id].document_id,
            content=by_id[chunk_id].content,
            score=scores[chunk_id],
            mode=SearchMode.HYBRID,
            metadata=by_id[chunk_id].metadata,
            source_scores=source_scores.get(chunk_id, {}),
        )
        for chunk_id in fused_ids
    ]
