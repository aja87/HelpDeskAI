from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_corpus import analyze_corpora, write_analysis
from scripts.compare_chunking import load_benchmark, write_comparison
from scripts.generate_golden_dataset import generate_golden_dataset


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


def test_golden_dataset_script_is_independent(tmp_path: Path) -> None:
    techqa_qa = [
        {
            "id": f"qa-{index}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "split": "train",
        }
        for index in range(80)
    ]
    bitext = [
        {
            "instruction": f"Request {index}",
            "response": f"Response {index}",
            "category": "ACCOUNT",
            "intent": "edit_account",
        }
        for index in range(30)
    ]
    qa_path = tmp_path / "qa.jsonl"
    bitext_path = tmp_path / "bitext.jsonl"
    output = tmp_path / "golden.jsonl"
    write_jsonl(qa_path, techqa_qa)
    write_jsonl(bitext_path, bitext)

    result = generate_golden_dataset(
        qa_path,
        bitext_path,
        output,
        seed=42,
        techqa_count=75,
        bitext_count=25,
        force=False,
    )

    assert result == output
    assert len(output.read_text(encoding="utf-8").splitlines()) == 100
