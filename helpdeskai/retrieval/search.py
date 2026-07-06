"""Public retrieval search API."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from helpdeskai.retrieval.corpus import ChunkRecord, load_chunks
from helpdeskai.retrieval.embeddings import SentenceTransformerEmbedder
from helpdeskai.retrieval.fusion import reciprocal_rank_fusion
from helpdeskai.retrieval.models import RetrievalConfig, SearchFilters, SearchMode, SearchResult
from helpdeskai.retrieval.sparse import SparseIndex


def _validate(top_k: int, mode: SearchMode | str) -> SearchMode:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    try:
        return SearchMode(mode)
    except ValueError as exc:
        raise ValueError("mode must be one of: dense, sparse, hybrid") from exc


class SearchEngine:
    """Search engine combining Qdrant dense search and BM25 sparse search."""

    def __init__(
        self,
        *,
        records: Sequence[ChunkRecord] | None = None,
        config: RetrievalConfig = RetrievalConfig(),
        embedder: SentenceTransformerEmbedder | None = None,
        qdrant_client=None,
        sparse_index: SparseIndex | None = None,
    ) -> None:
        self.config = config
        self.records = list(records) if records is not None else load_chunks(config.corpus_path)
        self.records_by_chunk_id = {record.chunk_id: record for record in self.records}
        self.embedder = embedder
        self.qdrant_client = qdrant_client
        self.sparse_index = sparse_index or SparseIndex(self.records)

    def _embedder(self) -> SentenceTransformerEmbedder:
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(self.config.model_name)
        return self.embedder

    def _qdrant(self):
        if self.qdrant_client is None:
            from qdrant_client import QdrantClient

            self.qdrant_client = QdrantClient(url=self.config.qdrant_url)
        return self.qdrant_client

    def _qdrant_filter(self, filters: SearchFilters | None):
        if filters is None:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        must = []
        if filters.product:
            must.append(FieldCondition(key="product", match=MatchValue(value=filters.product)))
        if filters.version:
            must.append(FieldCondition(key="versions", match=MatchValue(value=filters.version)))
        if filters.tenant:
            must.append(FieldCondition(key="tenant", match=MatchValue(value=filters.tenant)))
        if filters.date_from or filters.date_to:
            must.append(
                FieldCondition(
                    key="date",
                    range=Range(gte=filters.date_from, lte=filters.date_to),
                )
            )
        return Filter(must=must) if must else None

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant with dense embeddings."""
        vector = self._embedder().encode_query(query).tolist()
        response = self._qdrant().query_points(
            collection_name=self.config.collection_name,
            query=vector,
            query_filter=self._qdrant_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk_id = str(payload.get("chunk_id", hit.id))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(payload.get("document_id", "")),
                    content=str(payload.get("content", "")),
                    score=float(hit.score),
                    mode=SearchMode.DENSE,
                    metadata=dict(payload),
                    source_scores={"dense": float(hit.score)},
                )
            )
        return results

    def sparse_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Search BM25 index."""
        return self.sparse_index.search(query, top_k=top_k, filters=filters)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        mode: SearchMode | str = SearchMode.HYBRID,
    ) -> list[SearchResult]:
        """Search chunks using dense, sparse, or hybrid retrieval."""
        selected = _validate(top_k, mode)
        if selected is SearchMode.DENSE:
            return self.dense_search(query, top_k=top_k, filters=filters)
        if selected is SearchMode.SPARSE:
            return self.sparse_search(query, top_k=top_k, filters=filters)

        candidate_k = max(top_k * 4, 20)
        dense = self.dense_search(query, top_k=candidate_k, filters=filters)
        sparse = self.sparse_search(query, top_k=candidate_k, filters=filters)
        return reciprocal_rank_fusion([dense, sparse], k=self.config.rrf_k, top_k=top_k)


@lru_cache(maxsize=1)
def _default_engine() -> SearchEngine:
    return SearchEngine()


def search(
    query: str,
    top_k: int = 5,
    filters: SearchFilters | None = None,
    mode: SearchMode | str = SearchMode.HYBRID,
) -> list[SearchResult]:
    """Public retrieval function."""
    return _default_engine().search(query, top_k=top_k, filters=filters, mode=mode)
