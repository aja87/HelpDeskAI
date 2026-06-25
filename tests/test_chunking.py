from __future__ import annotations

import numpy as np
import pytest

from helpdeskai.corpus.chunking import (
    Chunk,
    WhitespaceTokenizer,
    build_chunk_records,
    fixed_size_chunks,
    recursive_chunks,
    semantic_chunks,
    stable_chunk_id,
)


class DeterministicEmbedder:
    model_name = "deterministic-test"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if "database" in text.lower():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


def test_fixed_size_chunks_respect_size_and_overlap() -> None:
    tokenizer = WhitespaceTokenizer()
    chunks = fixed_size_chunks(
        words(25),
        tokenizer,
        target_tokens=10,
        overlap_tokens=2,
    )

    assert [chunk.token_count for chunk in chunks] == [10, 10, 9, 1]
    assert chunks[0].content.split()[-2:] == chunks[1].content.split()[:2]


def test_recursive_chunks_preserve_boundaries_and_limits() -> None:
    tokenizer = WhitespaceTokenizer()
    text = f"{words(7)}.\n\n{words(7)}.\n\n{words(7)}."

    chunks = recursive_chunks(
        text,
        tokenizer,
        target_tokens=16,
        overlap_tokens=3,
    )

    assert len(chunks) == 2
    assert all(chunk.token_count <= 16 for chunk in chunks)
    assert chunks[0].content.split()[-3:] == chunks[1].content.split()[:3]


def test_semantic_chunks_split_on_topic_change() -> None:
    tokenizer = WhitespaceTokenizer()
    text = (
        f"Application server {words(60)}. "
        f"Application deployment {words(60)}. "
        f"Database connection {words(60)}. "
        f"Database schema {words(60)}."
    )

    chunks = semantic_chunks(
        text,
        tokenizer,
        DeterministicEmbedder(),
        min_tokens=50,
        max_tokens=200,
        breakpoint_percentile=75,
    )

    assert len(chunks) == 2
    assert "Application server" in chunks[0].content
    assert "Database connection" in chunks[1].content
    assert all(chunk.token_count <= 200 for chunk in chunks)


def test_stable_chunk_records_inherit_document_metadata() -> None:
    document = {
        "document_id": "techqa-doc",
        "text": "One two three",
        "metadata": {"category": {"value": "faq"}},
        "source_ids": ["Q1"],
        "splits": ["train"],
    }
    records = build_chunk_records(
        [document],
        lambda _: [Chunk("One two", 2, 0, "test")],
        tokenizer_name="test-tokenizer",
        strategy="test",
        strategy_params={"size": 2},
    )

    assert records[0]["document_id"] == "techqa-doc"
    assert records[0]["metadata"] is document["metadata"]
    assert records[0]["source_ids"] == ["Q1"]
    assert records[0]["chunk_id"] == stable_chunk_id("techqa-doc", 0, "One two")


def test_chunkers_validate_parameters() -> None:
    tokenizer = WhitespaceTokenizer()
    with pytest.raises(ValueError, match="overlap"):
        fixed_size_chunks("text", tokenizer, target_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError, match="percentile"):
        semantic_chunks(
            "Sentence.",
            tokenizer,
            DeterministicEmbedder(),
            breakpoint_percentile=101,
        )
