from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_corpus import analyze_corpora, write_analysis
from scripts.benchmark_retrieval import analyze_alignment, skipped_rows
from scripts.compare_chunking import (
    HashingSemanticEmbedder,
    build_semantic_embedder,
    build_tokenizer,
    load_benchmark,
    write_comparison,
)
from scripts.generate_corpus_quality_report import generate_report


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_corpus_analysis_script_writes_reports(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    write_jsonl(
        data_dir / "techqa/qa.jsonl",
        [
            {
                "id": "Q1",
                "split": "train",
                "question": "Question?",
                "answer": "Answer",
            }
        ],
    )
    write_jsonl(
        data_dir / "techqa/documents.jsonl",
        [{"id": "Q1", "split": "train", "document": "Document"}],
    )
    write_jsonl(
        data_dir / "bitext/tickets.jsonl",
        [
            {
                "instruction": "Help",
                "response": "Response",
                "category": "ACCOUNT",
                "intent": "edit_account",
            }
        ],
    )
    write_jsonl(
        data_dir / "msdialog/conversations.jsonl",
        [{"conversation_id": 1, "utterances": ["Hello", "Hi"]}],
    )

    summary, frames = analyze_corpora(data_dir)
    outputs = write_analysis(tmp_path / "reports", summary, frames)

    assert summary["overview"][0]["rows"] == 1
    assert all(path.is_file() for path in outputs)
    assert "TechQA duplicates" in outputs[1].read_text(encoding="utf-8")


def test_chunking_comparison_script_writes_chart(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "strategies": {
                    "fixed": {
                        "mean_tokens": 300,
                        "median_tokens": 320,
                        "runtime_seconds": 1,
                    },
                    "recursive": {
                        "mean_tokens": 280,
                        "median_tokens": 300,
                        "runtime_seconds": 2,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = load_benchmark(benchmark_path)
    output = write_comparison(metrics, tmp_path / "comparison.png")

    assert list(metrics.index) == ["fixed", "recursive"]
    assert output.is_file()


def test_chunking_comparison_rejects_missing_metrics(tmp_path: Path) -> None:
    dataframe = pd.DataFrame.from_dict(
        {"fixed": {"mean_tokens": 300}},
        orient="index",
    )

    with pytest.raises(ValueError, match="missing columns"):
        write_comparison(dataframe, tmp_path / "comparison.png")


def test_chunking_comparison_auto_uses_hashing_fallback(monkeypatch) -> None:
    def fail_bge_m3():
        raise RuntimeError("not enough memory")

    monkeypatch.setattr("scripts.compare_chunking.BgeM3Embedder", fail_bge_m3)

    embedder = build_semantic_embedder("auto")

    assert isinstance(embedder, HashingSemanticEmbedder)


def test_chunking_comparison_supports_whitespace_tokenizer() -> None:
    tokenizer = build_tokenizer("whitespace")

    assert tokenizer.name == "whitespace-v1"


def test_generate_corpus_quality_report_from_existing_chunks(monkeypatch, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_jsonl(
        processed / "documents.jsonl",
        [{"document_id": "doc-1", "text": "Document"}],
    )
    write_jsonl(
        processed / "chunks.jsonl",
        [
            {
                "chunk_id": "doc-1#0000",
                "document_id": "doc-1",
                "content": "Useful content",
                "content_hash": "hash-1",
                "token_count": 2,
                "metadata": {
                    "category": {"value": "faq"},
                    "product": {"value": "NovaCloud"},
                    "versions": {"value": ["1.0"]},
                    "date": {"value": "2024-01-01"},
                },
            }
        ],
    )

    def fake_evidently_report(chunks, output_path):
        output_path.write_text("<html>Evidently fixture</html>", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "scripts.generate_corpus_quality_report.generate_evidently_report",
        fake_evidently_report,
    )

    report, summary = generate_report(
        processed_dir=processed,
        report_path=tmp_path / "docs" / "corpus_quality_report.html",
        summary_path=tmp_path / "docs" / "corpus_quality_summary.json",
    )

    assert report.read_text(encoding="utf-8").startswith("<html>")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["chunks"] == 1
    assert payload["validated_documents"] == 1


def test_retrieval_benchmark_alignment_detects_missing_documents() -> None:
    cases = [
        {"document_id": "doc-1", "question": "q1"},
        {"document_id": "doc-2", "question": "q2"},
    ]

    alignment = analyze_alignment(cases, {"doc-1", "doc-3"})

    assert alignment["golden_cases"] == 2
    assert alignment["aligned_cases"] == 1
    assert alignment["missing_unique_documents"] == 1
    assert alignment["missing_sample"] == ["doc-2"]
    assert skipped_rows()[0]["recall@5"] == "n/a"


