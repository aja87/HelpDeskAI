from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from helpdeskai.retrieval.corpus import ChunkRecord, flatten_chunk, matches_filters
from helpdeskai.retrieval.fusion import reciprocal_rank_fusion
from helpdeskai.retrieval.models import SearchFilters, SearchMode, SearchResult
from helpdeskai.retrieval.search import SearchEngine
from helpdeskai.retrieval.sparse import SparseIndex, tokenize


class FakeEmbedder:
    model_name = "fake"

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeQdrant:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits

    def query_points(self, **kwargs):
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=hit["chunk_id"], score=hit["score"], payload=hit)
                for hit in self.hits[: kwargs["limit"]]
            ]
        )


def record(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "doc-1",
    product: str | None = "NovaDesk",
    versions: list[str] | None = None,
    published: str | None = "2024-01-15",
    tenant: str = "novacloud-core",
) -> ChunkRecord:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": content,
        "product": product,
        "versions": versions or ["3.2"],
        "date": published,
        "category": "faq",
        "tenant": tenant,
    }
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        metadata=metadata,
        payload=metadata,
    )


def result(chunk_id: str, score: float, mode: SearchMode) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content {chunk_id}",
        score=score,
        mode=mode,
        metadata={"chunk_id": chunk_id},
        source_scores={mode.value: score},
    )


def test_sparse_tokenization_and_ranking_returns_lexical_match() -> None:
    records = [
        record("auth", "Configure SAML authentication and identity federation"),
        record("billing", "Download invoices and billing history"),
        record("storage", "Backup retention and restore policy"),
    ]
    index = SparseIndex(records)

    assert tokenize("SAML 2.0 /API") == ["saml", "2", "0", "api"]
    hits = index.search("SAML identity", top_k=1)

    assert hits[0].chunk_id == "auth"
    assert hits[0].mode is SearchMode.SPARSE
    assert hits[0].source_scores["sparse"] > 0


def test_rrf_combines_rankings_deterministically() -> None:
    dense = [
        result("a", 0.9, SearchMode.DENSE),
        result("b", 0.8, SearchMode.DENSE),
    ]
    sparse = [
        result("b", 4.0, SearchMode.SPARSE),
        result("c", 3.0, SearchMode.SPARSE),
    ]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].mode is SearchMode.HYBRID
    assert fused[0].source_scores == {"dense": 0.8, "sparse": 4.0}


def test_metadata_filters_include_and_exclude_product_version_date_and_tenant() -> None:
    item = record("one", "content")

    assert matches_filters(
        item,
        SearchFilters(
            product="NovaDesk",
            version="3.2",
            date_from="2024-01-01",
            date_to="2024-12-31",
            tenant="novacloud-core",
        ),
    )
    assert not matches_filters(item, SearchFilters(product="Other"))
    assert not matches_filters(item, SearchFilters(version="9.9"))
    assert not matches_filters(item, SearchFilters(date_from="2025-01-01"))
    assert not matches_filters(item, SearchFilters(tenant="novacloud-ops"))


def test_search_validates_mode_and_top_k() -> None:
    engine = SearchEngine(records=[record("one", "content")], qdrant_client=FakeQdrant([]))

    with pytest.raises(ValueError, match="top_k"):
        engine.search("query", top_k=0)
    with pytest.raises(ValueError, match="mode"):
        engine.search("query", mode="unknown")


def test_hybrid_search_returns_fused_results_with_source_scores() -> None:
    records = [
        record("dense-only", "unrelated dense payload", document_id="doc-dense"),
        record("shared", "saml identity shared lexical", document_id="doc-shared"),
    ]
    qdrant = FakeQdrant(
        [
            records[0].payload | {"score": 0.99},
            records[1].payload | {"score": 0.8},
        ]
    )
    engine = SearchEngine(
        records=records,
        embedder=FakeEmbedder(),
        qdrant_client=qdrant,
    )

    hits = engine.search("saml identity", top_k=2, mode="hybrid")

    assert hits[0].mode is SearchMode.HYBRID
    assert any(hit.chunk_id == "shared" for hit in hits)
    shared = next(hit for hit in hits if hit.chunk_id == "shared")
    assert set(shared.source_scores) == {"dense", "sparse"}


def test_flatten_chunk_preserves_payload_and_metadata() -> None:
    flattened = flatten_chunk(
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "position": 2,
            "content": "Support content",
            "content_hash": "abc",
            "source_ids": ["Q1"],
            "splits": ["train"],
            "chunking": {"strategy": "recursive"},
            "metadata": {
                "title": "Reset password",
                "product": {"value": "NovaDesk"},
                "versions": {"value": ["3.2"]},
                "date": {"value": "2024-02-01"},
                "category": {"value": "faq"},
            },
        }
    )

    assert flattened.chunk_id == "chunk-1"
    assert flattened.document_id == "doc-1"
    assert flattened.content == "Support content"
    assert flattened.payload["product"] == "NovaDesk"
    assert flattened.payload["versions"] == ["3.2"]
    assert flattened.payload["date"] == "2024-02-01"
    assert flattened.payload["source_ids"] == ["Q1"]
    assert flattened.payload["raw_metadata"]["title"] == "Reset password"
