from __future__ import annotations

import logging
import math
import re
import time

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from .config import RetrievalConfig
from .io_utils import read_jsonl, write_json

SearchMode = Literal["dense", "sparse", "hybrid"]


class EmbeddingModel(Protocol):
    """Protocol for text embedding providers."""

    def encode(self, text: str, *, normalize_embeddings: bool = True) -> list[float]:
        """Encode an input query into a dense vector."""


class SimpleBM25:
    """Small BM25 implementation for deterministic local sparse retrieval."""

    def __init__(self, tokenized_corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._corpus = tokenized_corpus
        self._k1 = k1
        self._b = b
        self._doc_count = len(tokenized_corpus)
        self._doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self._doc_lengths = [len(doc) for doc in tokenized_corpus]
        self._avg_doc_len = (
            sum(self._doc_lengths) / self._doc_count if self._doc_count else 0.0
        )
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_freq_by_term: Counter[str] = Counter()
        for doc in self._corpus:
            for term in set(doc):
                doc_freq_by_term[term] += 1

        idf: dict[str, float] = {}
        for term, freq in doc_freq_by_term.items():
            numerator = self._doc_count - freq + 0.5
            denominator = freq + 0.5
            idf[term] = math.log(1 + (numerator / denominator))
        return idf

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0 for _ in self._corpus]
        if not query_tokens or self._avg_doc_len == 0.0:
            return scores

        for index, term_freqs in enumerate(self._doc_freqs):
            doc_len = self._doc_lengths[index]
            length_norm = self._k1 * (1 - self._b + self._b * (doc_len / self._avg_doc_len))
            total = 0.0
            for term in query_tokens:
                term_freq = term_freqs.get(term, 0)
                if term_freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                numerator = term_freq * (self._k1 + 1)
                denominator = term_freq + length_norm
                total += idf * (numerator / denominator)
            scores[index] = total
        return scores


@dataclass(slots=True)
class SearchFilters:
    """Supported metadata filters for retrieval."""

    source: str | None = None
    product: str | None = None
    version: str | None = None
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None


@dataclass(slots=True)
class SearchHit:
    """Single retrieval result row."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    source: str
    product: str
    version: str
    category: str
    date: str


def tokenize(text: str) -> list[str]:
    """Lowercase + split for BM25 tokenization."""

    return [token for token in re.split(r"\W+", text.lower()) if token]


def rrf_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Fuse rankings with Reciprocal Rank Fusion."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])[:top_n]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@k over chunk identifiers."""

    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute reciprocal rank for the first relevant hit."""

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def p95(values: list[float]) -> float:
    """Compute deterministic p95 for latency values."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(0.95 * len(ordered)) - 1, 0)
    return ordered[index]


def _build_qdrant_filter(filters: SearchFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None

    conditions: list[dict[str, Any]] = []
    if filters.source:
        conditions.append({"key": "source", "match": {"value": filters.source}})
    if filters.product:
        conditions.append({"key": "product", "match": {"value": filters.product}})
    if filters.version:
        conditions.append({"key": "version", "match": {"value": filters.version}})
    if filters.category:
        conditions.append({"key": "category", "match": {"value": filters.category}})

    if filters.date_from or filters.date_to:
        conditions.append(
            {
                "key": "date",
                "range": {
                    "gte": filters.date_from,
                    "lte": filters.date_to,
                },
            }
        )

    return {"must": conditions} if conditions else None


def _matches_filters(row: dict[str, Any], filters: SearchFilters | None) -> bool:
    if filters is None:
        return True

    if filters.source and row.get("source") != filters.source:
        return False
    if filters.product and row.get("product") != filters.product:
        return False
    if filters.version and row.get("version") != filters.version:
        return False
    if filters.category and row.get("category") != filters.category:
        return False

    row_date = str(row.get("date", ""))
    if filters.date_from and row_date and row_date < filters.date_from:
        return False
    if filters.date_to and row_date and row_date > filters.date_to:
        return False

    return True


class RetrievalEngine:
    """Search and benchmark engine for dense, sparse, and hybrid retrieval."""

    def __init__(
        self,
        config: RetrievalConfig,
        *,
        qdrant_client: Any | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        config.validate()
        self.config = config
        if qdrant_client is None:
            from helpdeskai.indexing.qdrant_store import create_qdrant_client

            self.qdrant_client = create_qdrant_client(
                config.qdrant_url,
                api_key=config.qdrant_api_key,
            )
        else:
            self.qdrant_client = qdrant_client

        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(
                config.embedding_model,
                trust_remote_code=True,
            )
        else:
            self.embedding_model = embedding_model

        self._chunks = read_jsonl(config.chunks_path)
        if not self._chunks:
            raise ValueError(f"No chunks found in {config.chunks_path}")

        self._chunk_ids = [str(row.get("chunk_id", "")) for row in self._chunks]
        self._chunk_by_id = {str(row.get("chunk_id", "")): row for row in self._chunks}
        self._tokens = [tokenize(str(row.get("text", ""))) for row in self._chunks]
        self._bm25 = SimpleBM25(self._tokens)
        self._doc_to_chunk_ids = self._build_doc_index(self._chunks)

    @staticmethod
    def _build_doc_index(chunks: list[dict[str, Any]]) -> dict[str, set[str]]:
        by_doc: dict[str, set[str]] = {}
        for row in chunks:
            doc_id = str(row.get("doc_id", ""))
            chunk_id = str(row.get("chunk_id", ""))
            if not doc_id or not chunk_id:
                continue
            by_doc.setdefault(doc_id, set()).add(chunk_id)
        return by_doc

    @staticmethod
    def _to_hit(row: dict[str, Any], score: float) -> SearchHit:
        return SearchHit(
            chunk_id=str(row.get("chunk_id", "")),
            doc_id=str(row.get("doc_id", "")),
            score=float(score),
            text=str(row.get("text", "")),
            source=str(row.get("source", "")),
            product=str(row.get("product", "")),
            version=str(row.get("version", "")),
            category=str(row.get("category", "")),
            date=str(row.get("date", "")),
        )

    def _dense_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None,
    ) -> list[SearchHit]:
        query_vector = self.embedding_model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )
        query_filter = _build_qdrant_filter(filters)

        hits = self.qdrant_client.search(
            collection_name=self.config.collection_name,
            query_vector=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        result_hits: list[SearchHit] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            if "chunk_id" not in payload:
                continue
            result_hits.append(self._to_hit(payload, score=float(hit.score or 0.0)))
        return result_hits

    def _sparse_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None,
    ) -> list[SearchHit]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda index: -scores[index])

        results: list[SearchHit] = []
        for index in ranked_indices:
            row = self._chunks[index]
            if not _matches_filters(row, filters):
                continue
            results.append(self._to_hit(row, score=float(scores[index])))
            if len(results) >= top_k:
                break
        return results

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        mode: SearchMode | str = "hybrid",
    ) -> list[SearchHit]:
        """Search chunks with dense, sparse, or hybrid retrieval mode."""

        top_k = top_k or self.config.top_k
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("mode must be one of ['dense', 'sparse', 'hybrid']")

        if mode == "dense":
            return self._dense_search(query, top_k=top_k, filters=filters)

        if mode == "sparse":
            return self._sparse_search(query, top_k=top_k, filters=filters)

        candidate_k = max(top_k * 3, top_k)
        dense_hits = self._dense_search(query, top_k=candidate_k, filters=filters)
        sparse_hits = self._sparse_search(query, top_k=candidate_k, filters=filters)

        fused = rrf_fusion(
            [
                [hit.chunk_id for hit in dense_hits],
                [hit.chunk_id for hit in sparse_hits],
            ],
            k=self.config.rrf_k,
            top_n=top_k,
        )

        fused_hits: list[SearchHit] = []
        for chunk_id, score in fused:
            row = self._chunk_by_id.get(chunk_id)
            if row is None:
                continue
            fused_hits.append(self._to_hit(row, score=score))
        return fused_hits

    def _relevance_from_golden_row(self, row: dict[str, Any]) -> set[str]:
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id and doc_id in self._doc_to_chunk_ids:
            return self._doc_to_chunk_ids[doc_id]

        expected_answer = str(row.get("expected_answer", "")).strip()
        if not expected_answer:
            return set()

        ranked = self._sparse_search(
            expected_answer,
            top_k=self.config.benchmark_relevance_top_n,
            filters=None,
        )
        return {hit.chunk_id for hit in ranked}

    def benchmark(
        self,
        *,
        golden_path: Path | None = None,
        sample_size: int | None = None,
        top_k: int | None = None,
        save_path: Path | None = None,
    ) -> dict[str, Any]:
        """Run retrieval benchmark across dense, sparse, and hybrid modes."""

        benchmark_top_k = top_k or max(self.config.top_k, 10)
        target_size = sample_size or self.config.benchmark_sample_size
        golden_rows = read_jsonl(self.config.golden_path if golden_path is None else golden_path)
        selected_rows = golden_rows[:target_size]

        report: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config": {
                "collection_name": self.config.collection_name,
                "embedding_model": self.config.embedding_model,
                "sample_size": len(selected_rows),
                "top_k": benchmark_top_k,
                "rrf_k": self.config.rrf_k,
            },
            "metrics": {},
        }

        modes: tuple[SearchMode, ...] = ("dense", "sparse", "hybrid")
        for mode in modes:
            recalls_at_5: list[float] = []
            recalls_at_10: list[float] = []
            mrr_values: list[float] = []
            latencies_ms: list[float] = []
            rows_with_relevance = 0

            for row in selected_rows:
                query = str(row.get("question", "")).strip()
                if not query:
                    continue

                relevant = self._relevance_from_golden_row(row)
                if relevant:
                    rows_with_relevance += 1

                start = time.perf_counter()
                results = self.search(query, top_k=benchmark_top_k, mode=mode)
                latency_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(latency_ms)

                retrieved_chunk_ids = [hit.chunk_id for hit in results]
                recalls_at_5.append(recall_at_k(retrieved_chunk_ids, relevant, 5))
                recalls_at_10.append(recall_at_k(retrieved_chunk_ids, relevant, 10))
                mrr_values.append(reciprocal_rank(retrieved_chunk_ids, relevant))

            count = len(recalls_at_5)
            report["metrics"][mode] = {
                "queries_evaluated": count,
                "queries_with_relevance": rows_with_relevance,
                "recall_at_5": sum(recalls_at_5) / count if count else 0.0,
                "recall_at_10": sum(recalls_at_10) / count if count else 0.0,
                "mrr": sum(mrr_values) / count if count else 0.0,
                "latency_p95_ms": p95(latencies_ms),
                "latency_avg_ms": sum(latencies_ms) / count if count else 0.0,
            }

        output_path = self.config.benchmark_path if save_path is None else save_path
        write_json(output_path, report)
        return report


def run_retrieval_core(
    config: RetrievalConfig,
    *,
    query: str,
    top_k: int | None = None,
    mode: SearchMode | str | None = None,
    filters: SearchFilters | None = None,
) -> dict[str, Any]:
    """Run retrieval search for developer validation."""

    logging.info("Starting retrieval workflow with config: %s", asdict(config))
    engine = RetrievalEngine(config)
    active_mode = mode or config.mode
    results = engine.search(query=query, top_k=top_k, mode=active_mode, filters=filters)

    payload = {
        "query": query,
        "mode": active_mode,
        "top_k": top_k or config.top_k,
        "filters": asdict(filters) if filters else {},
        "results": [asdict(hit) for hit in results],
    }
    return payload


def run_benchmark_core(
    config: RetrievalConfig,
    *,
    sample_size: int | None = None,
    top_k: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run retrieval benchmark and persist report."""

    logging.info("Starting retrieval benchmark with config: %s", asdict(config))
    engine = RetrievalEngine(config)
    return engine.benchmark(sample_size=sample_size, top_k=top_k, save_path=output_path)
