from __future__ import annotations

from helpdeskai.retrieval.workflow import (
    SearchFilters,
    _build_qdrant_filter,
    _matches_filters,
    p95,
    recall_at_k,
    reciprocal_rank,
    rrf_fusion,
    tokenize,
)


def test_tokenize_splits_on_non_alphanumeric() -> None:
    assert tokenize("SSL Error: AMQ9716 in MQ 7.0.1.0") == [
        "ssl",
        "error",
        "amq9716",
        "in",
        "mq",
        "7",
        "0",
        "1",
        "0",
    ]


def test_rrf_fusion_combines_rankings_deterministically() -> None:
    fused = rrf_fusion([["C1", "C2", "C3"], ["C2", "C4", "C1"]], k=60, top_n=3)
    assert [chunk_id for chunk_id, _ in fused] == ["C2", "C1", "C4"]


def test_recall_and_reciprocal_rank_metrics() -> None:
    retrieved = ["C10", "C20", "C30", "C40"]
    relevant = {"C30", "C99"}

    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 1.0 / 3.0


def test_build_qdrant_filter_includes_requested_fields() -> None:
    filters = SearchFilters(source="techqa", product="MQ", category="troubleshooting")
    qfilter = _build_qdrant_filter(filters)

    assert qfilter is not None
    assert len(qfilter["must"]) == 3


def test_matches_filters_rejects_non_matching_metadata() -> None:
    row = {
        "source": "techqa",
        "product": "MQ",
        "version": "9.3",
        "category": "troubleshooting",
        "date": "2024-01-01",
    }
    filters = SearchFilters(source="bitext")
    assert not _matches_filters(row, filters)


def test_p95_returns_high_percentile_value() -> None:
    values = [10.0, 25.0, 15.0, 200.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    assert p95(values) == 200.0
