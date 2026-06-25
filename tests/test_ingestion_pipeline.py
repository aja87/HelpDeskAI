from __future__ import annotations

import json
from pathlib import Path

from helpdeskai.ingestion.pipeline import (
    IngestionConfig,
    benchmark_chunking_strategies,
    build_golden_dataset,
    chunk_documents,
    normalize_text,
    prepare_msdialog_conversations,
    prepare_qa_pairs,
    prepare_techqa_documents,
)


def test_prepare_techqa_documents_deduplicates_and_enriches() -> None:
    raw_documents = [
        {
            "doc_id": "A1",
            "text": "IBM WebSphere MQ 7.0.1.0 TECHNOTE (TROUBLESHOOTING)<p>Fix issue</p>",
            "title": "",
            "product": "",
            "version": "",
            "category": "",
            "date": "",
        },
        {
            "doc_id": "A2",
            "text": "IBM WebSphere MQ 7.0.1.0 TECHNOTE (TROUBLESHOOTING)<p>Fix issue</p>",
            "title": "",
            "product": "",
            "version": "",
            "category": "",
            "date": "",
        },
    ]

    prepared = prepare_techqa_documents(raw_documents)

    assert len(prepared) == 1
    assert prepared[0]["version"] == "7.0.1.0"
    assert prepared[0]["category"] == "troubleshooting"
    assert "Fix issue" in prepared[0]["text"]


def test_prepare_qa_pairs_normalizes_and_deduplicates_questions() -> None:
    rows = [
        {"question_id": "1", "question": "<b>Hello?</b>", "answer": "Yes", "doc_id": "D1"},
        {"question_id": "2", "question": "Hello?", "answer": "Yes again", "doc_id": "D2"},
    ]

    prepared = prepare_qa_pairs(rows, source="techqa")

    assert len(prepared) == 1
    assert prepared[0]["question"] == "Hello?"


def test_prepare_msdialog_conversations_uses_checksum_fallback_id() -> None:
    rows = [{"conversation_id": "", "text": "Question [SEP] Answer", "final_answer": "Answer"}]

    prepared = prepare_msdialog_conversations(rows)

    assert len(prepared) == 1
    assert prepared[0]["conversation_id"]
    assert "Question" in prepared[0]["text"]


def test_benchmark_and_chunk_documents_return_consistent_output() -> None:
    documents = [
        {
            "doc_id": "DOC1",
            "title": "Doc 1",
            "text": normalize_text("Paragraph one.\n\nParagraph two with more content.\n\nParagraph three."),
            "product": "IBM MQ",
            "version": "7.0.1.0",
            "category": "troubleshooting",
            "date": "",
            "source": "techqa",
            "checksum": "x",
            "char_count": 80,
            "word_count": 12,
        }
    ]
    config = IngestionConfig(chunk_sample_size=1)

    report = benchmark_chunking_strategies(documents, config)
    chunks = chunk_documents(documents, report["recommended_strategy"], config)

    assert report["recommended_strategy"] in {"fixed", "recursive", "semantic"}
    assert len(chunks) >= 1
    assert chunks[0]["doc_id"] == "DOC1"


def test_build_golden_dataset_writes_requested_size(tmp_path: Path) -> None:
    techqa_pairs = [
        {"question_id": f"t{i}", "question": f"tech question {i}", "answer": "a", "doc_id": "d", "intent": "", "category": "", "source": "techqa"}
        for i in range(10)
    ]
    bitext_pairs = [
        {"question_id": f"b{i}", "question": f"bitext question {i}", "answer": "a", "doc_id": "", "intent": "x", "category": "account", "source": "bitext"}
        for i in range(10)
    ]
    output_path = tmp_path / "golden.jsonl"

    golden_rows = build_golden_dataset(techqa_pairs, bitext_pairs, output_path, target_size=12, seed=42)

    assert len(golden_rows) == 12
    written_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(written_lines) == 12
    first_row = json.loads(written_lines[0])
    assert "expected_answer" in first_row