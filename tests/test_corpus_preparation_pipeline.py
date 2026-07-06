from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault(
    "PREFECT_HOME",
    str(Path(".pytest_cache/prefect").resolve()),
)

import helpdeskai.ingestion.pipeline as pipeline  # noqa: E402
from helpdeskai.ingestion.exceptions import TechQAIngestionError  # noqa: E402


def canonical_documents() -> list[dict]:
    metadata = {
        name: {"value": value, "method": "fixture", "confidence": 1.0}
        for name, value in {
            "category": "faq",
            "product": "Product",
            "versions": ["1.0"],
            "date": "2024-01-01",
        }.items()
    }
    return [
        {
            "document_id": "doc-001",
            "content_hash": "hash-001",
            "text": "Normalized support documentation",
            "metadata": metadata,
            "source_ids": ["Q1"],
            "splits": ["train"],
            "source_record_count": 1,
            "normalization_version": "1.0",
        }
    ]


def final_chunks() -> list[dict]:
    metadata = canonical_documents()[0]["metadata"]
    return [
        {
            "chunk_id": "doc-001#0000-hash",
            "document_id": "doc-001",
            "position": 0,
            "content": "Normalized support documentation",
            "content_hash": "hash",
            "token_count": 4,
            "metadata": metadata,
            "source_ids": ["Q1"],
            "splits": ["train"],
            "chunking": {
                "version": "recursive-1.0",
                "strategy": "recursive",
                "parameters": {"target_tokens": 384, "overlap_tokens": 64},
                "tokenizer": "test",
            },
        }
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_quality_report_task_publishes_report_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "docs" / "corpus_quality_report.html"

    def fake_report(chunks: list[dict], path: Path) -> Path:
        path.write_text("<html>Evidently fixture</html>", encoding="utf-8")
        return path

    monkeypatch.setattr(pipeline, "generate_evidently_report", fake_report)

    result = pipeline.quality_report_task.fn(
        final_chunks(),
        {"doc-001"},
        report,
        force=False,
    )
    summary = json.loads(
        (report.parent / "corpus_quality_summary.json").read_text(encoding="utf-8")
    )

    assert result == report
    assert report.is_file()
    assert summary["chunks"] == 1
    assert summary["unique_chunk_ids"] == 1

    with pytest.raises(TechQAIngestionError, match="--force"):
        pipeline.quality_report_task.fn(
            final_chunks(),
            {"doc-001"},
            report,
            force=False,
        )


def test_corpus_preparation_flow_skip_existing_is_idempotent(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    report = tmp_path / "docs" / "corpus_quality_report.html"
    for path in (
        processed / "documents.jsonl",
        processed / "chunks.jsonl",
        processed / "manifest.json",
        report,
        report.parent / "corpus_quality_summary.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    result = pipeline.corpus_preparation_flow.fn(
        raw_dir=tmp_path / "raw",
        processed_dir=processed,
        report_path=report,
        skip_existing=True,
    )

    assert result == report


def test_corpus_preparation_flow_rejects_partial_skip_existing(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "documents.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(TechQAIngestionError, match="incomplete"):
        pipeline.corpus_preparation_flow.fn(
            raw_dir=tmp_path / "raw",
            processed_dir=processed,
            report_path=tmp_path / "quality.html",
            skip_existing=True,
        )
