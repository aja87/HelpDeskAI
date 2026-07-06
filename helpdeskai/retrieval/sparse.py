"""BM25 sparse retrieval over processed chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from helpdeskai.retrieval.corpus import ChunkRecord, filter_chunks
from helpdeskai.retrieval.models import SearchFilters, SearchMode, SearchResult

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase words and numeric references."""
    return TOKEN_RE.findall(text.casefold())


class _FallbackBM25:
    """Small BM25 implementation used when rank-bm25 is not installed."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = [list(document) for document in corpus]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(document) for document in self.corpus) / max(len(self.corpus), 1)
        doc_freq: Counter[str] = Counter()
        for document in self.corpus:
            doc_freq.update(set(document))
        doc_count = len(self.corpus)
        self.idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = []
        for document in self.corpus:
            frequencies = Counter(document)
            doc_len = len(document)
            score = 0.0
            for term in query_tokens:
                freq = frequencies.get(term, 0)
                if not freq:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)


def _bm25(tokenized: Sequence[Sequence[str]]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return _FallbackBM25(tokenized)
    return BM25Okapi(tokenized)


class SparseIndex:
    """In-memory BM25 index with filter-aware ranking."""

    def __init__(self, records: Sequence[ChunkRecord]) -> None:
        self.records = list(records)
        self._tokenized = [tokenize(record.content) for record in self.records]
        self._bm25 = _bm25(self._tokenized)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Return BM25-ranked chunks."""
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        allowed = {record.chunk_id for record in filter_chunks(self.records, filters)}
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if self.records[index].chunk_id in allowed
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        results = []
        for index, score in ranked[:top_k]:
            record = self.records[index]
            results.append(
                SearchResult(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    content=record.content,
                    score=score,
                    mode=SearchMode.SPARSE,
                    metadata=record.metadata,
                    source_scores={"sparse": score},
                )
            )
        return results
