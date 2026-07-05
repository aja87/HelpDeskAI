from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpdeskai.corpus.benchmark import (
    compare_strategies,
    deterministic_document_sample,
    write_benchmark,
)
from helpdeskai.corpus.chunking import Chunk
from helpdeskai.ingestion.quality import (
    CorpusQualityError,
    generate_evidently_report,
    validate_chunks,
)


def documents(count: int = 60) -> list[dict]:
    return [
        {"document_id": f"doc-{index:03d}", "text": "word " * (index + 5)} for index in range(count)
    ]


def test_document_sample_is_deterministic_and_length_stratified() -> None:
    first = deterministic_document_sample(documents(), sample_size=50, seed=42)
    second = deterministic_document_sample(documents(), sample_size=50, seed=42)

    assert [record["document_id"] for record in first] == [
        record["document_id"] for record in second
    ]
    assert len(first) == 50
    lengths = [len(record["text"]) for record in first]
    assert min(lengths) < 100
    assert max(lengths) > 200


def test_benchmark_calculates_metrics_and_writes_artifacts(tmp_path: Path) -> None:
    sample = documents(3)
    benchmark = compare_strategies(
        sample,
        {
            "one": lambda text: [Chunk(text, len(text.split()), 0, "one")],
            "two": lambda text: [
                Chunk(text, len(text.split()), 0, "two"),
                Chunk("duplicate", 1, 1, "two"),
            ],
        },
    )
    json_path, markdown_path = write_benchmark(tmp_path, benchmark)

    assert benchmark["strategies"]["one"]["chunks"] == 3
    assert benchmark["strategies"]["two"]["duplicate_chunks"] == 2
    assert json.loads(json_path.read_text())["strategies"]["one"]["documents"] == 3
    assert "Recursive chunking" in markdown_path.read_text(encoding="utf-8")


def valid_chunk(chunk_id: str = "doc#0", document_id: str = "doc") -> dict:
    metadata = {
        name: {"value": value}
        for name, value in {
            "category": "faq",
            "product": "Product",
            "versions": ["1.0"],
            "date": "2024-01-01",
        }.items()
    }
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": "Useful content",
        "content_hash": chunk_id,
        "token_count": 20,
        "metadata": metadata,
    }


def test_quality_contract_and_failures() -> None:
    report = validate_chunks([valid_chunk()], {"doc"})
    assert report["unique_chunk_ids"] == 1
    assert report["metadata_coverage"]["product"] == 1

    with pytest.raises(CorpusQualityError, match="duplicate"):
        validate_chunks([valid_chunk(), valid_chunk()], {"doc"})
    with pytest.raises(CorpusQualityError, match="unknown document"):
        validate_chunks([valid_chunk(document_id="missing")], {"doc"})
    invalid = valid_chunk()
    invalid["token_count"] = 513
    with pytest.raises(CorpusQualityError, match="token_count"):
        validate_chunks([invalid], {"doc"})


def test_evidently_quality_report_is_html(tmp_path: Path) -> None:
    path = generate_evidently_report(
        [valid_chunk()],
        tmp_path / "quality.html",
    )

    html = path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "evidently" in html.lower()
