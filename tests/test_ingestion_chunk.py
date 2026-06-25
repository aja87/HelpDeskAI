from __future__ import annotations

from helpdeskai.ingestion.chunk import WhitespaceTokenizer, chunk_documents
from helpdeskai.ingestion.dedup import deduplicate_chunks


def test_selected_chunk_stage_inherits_metadata_and_respects_limits() -> None:
    metadata = {
        name: {"value": value}
        for name, value in {
            "category": "faq",
            "product": "Product",
            "versions": ["1.0"],
            "date": "2024-01-01",
        }.items()
    }
    documents = [
        {
            "document_id": "doc-1",
            "text": "support information " * 50,
            "metadata": metadata,
            "source_ids": ["Q1"],
            "splits": ["train"],
        }
    ]

    chunks = chunk_documents(
        documents,
        WhitespaceTokenizer(),
        target_tokens=20,
        overlap_tokens=4,
    )

    assert len(chunks) > 1
    assert all(chunk["token_count"] <= 20 for chunk in chunks)
    assert all(chunk["document_id"] == "doc-1" for chunk in chunks)
    assert chunks[0]["metadata"] is metadata


def test_chunk_deduplication_keeps_first_occurrence() -> None:
    chunks = [
        {"chunk_id": "one", "content": "same", "content_hash": "hash"},
        {"chunk_id": "two", "content": "same", "content_hash": "hash"},
    ]

    unique = deduplicate_chunks(chunks)

    assert [chunk["chunk_id"] for chunk in unique] == ["one"]
