from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from helpdeskai.corpus.analysis import (
    CORPUS_PATHS,
    add_conversation_metrics,
    add_text_length,
    dataframe_overview,
    document_duplicate_groups,
    find_project_root,
    load_corpora,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_corpora_and_basic_analysis(tmp_path: Path) -> None:
    records = {
        "techqa_qa": [
            {
                "id": "q1",
                "split": "train",
                "document": "Document",
                "question": "Question?",
                "answer": "Answer",
            }
        ],
        "techqa_documents": [{"id": "q1", "split": "train", "document": "Document"}],
        "bitext": [
            {
                "flags": "B",
                "instruction": "Help",
                "category": "ACCOUNT",
                "intent": "edit_account",
                "response": "Response",
            }
        ],
        "msdialog": [{"conversation_id": 1, "utterances": ["Hello", "Hi"]}],
    }
    for name, relative_path in CORPUS_PATHS.items():
        write_jsonl(tmp_path / relative_path, records[name])

    corpora = load_corpora(tmp_path)
    overview = dataframe_overview(corpora)
    documents = add_text_length(corpora["techqa_documents"], "document", "document_length")
    conversations = add_conversation_metrics(corpora["msdialog"])

    assert set(corpora) == set(CORPUS_PATHS)
    assert overview.loc["techqa_qa", "rows"] == 1
    assert documents.loc[0, "document_length"] == 8
    assert conversations.loc[0, "turn_count"] == 2


def test_load_corpora_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_corpus.py"):
        load_corpora(tmp_path)


def test_find_project_root(tmp_path: Path) -> None:
    nested = tmp_path / "scripts" / "nested"
    nested.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert find_project_root(nested) == tmp_path


def test_analysis_helpers_validate_columns() -> None:
    with pytest.raises(KeyError, match="document"):
        add_text_length(pd.DataFrame({"other": ["text"]}), "document", "length")

    conversations = add_conversation_metrics(pd.DataFrame([{"messages": ["one"]}]))
    assert conversations.loc[0, "turn_count"] == 1


def test_document_duplicate_groups_compares_content_not_full_rows() -> None:
    dataframe = pd.DataFrame(
        [
            {"id": "Q1", "split": "train", "document": "Same document"},
            {"id": "Q2", "split": "validation", "document": "Same document"},
            {"id": "Q3", "split": "train", "document": "Unique document"},
        ]
    )

    groups = document_duplicate_groups(dataframe)

    assert len(groups) == 1
    assert groups.loc[0, "duplicate_count"] == 2
    assert groups.loc[0, "duplicate_rows_removed"] == 1
    assert groups.loc[0, "source_ids"] == ["Q1", "Q2"]
    assert groups.loc[0, "splits"] == ["train", "validation"]
    assert bool(groups.loc[0, "cross_split"]) is True
