"""Reusable helpers for standalone corpus analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

CORPUS_PATHS = {
    "techqa_qa": Path("techqa/qa.jsonl"),
    "techqa_documents": Path("techqa/documents.jsonl"),
    "bitext": Path("bitext/tickets.jsonl"),
    "msdialog": Path("msdialog/conversations.jsonl"),
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent directory containing pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Run the script from the HelpDeskAI repository."
    )


def load_corpora(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all expected JSONL corpus exports."""
    data_dir = data_dir.resolve()
    missing = [
        (data_dir / relative_path)
        for relative_path in CORPUS_PATHS.values()
        if not (data_dir / relative_path).is_file()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Corpus files are missing:\n"
            f"{formatted}\n"
            "Run `uv run python scripts/download_corpus.py` from the project root."
        )

    return {
        name: pd.read_json(data_dir / relative_path, lines=True)
        for name, relative_path in CORPUS_PATHS.items()
    }


def dataframe_overview(dataframes: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact shape and completeness overview."""
    rows = []
    for name, dataframe in dataframes.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "missing_values": int(dataframe.isna().sum().sum()),
                "duplicate_rows": int(dataframe.astype(str).duplicated().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def document_duplicate_groups(
    dataframe: pd.DataFrame,
    content_column: str = "document",
) -> pd.DataFrame:
    """Summarize duplicate document content and its source aliases."""
    if content_column not in dataframe:
        raise KeyError(f"Column '{content_column}' is missing")

    working = dataframe.copy()
    working["_normalized_content"] = working[content_column].fillna("").astype(str).str.strip()
    duplicate_rows = working[working.duplicated("_normalized_content", keep=False)].copy()
    if duplicate_rows.empty:
        return pd.DataFrame(
            columns=[
                "duplicate_count",
                "duplicate_rows_removed",
                "source_ids",
                "splits",
                "cross_split",
                "preview",
            ]
        )

    groups = []
    for content, group in duplicate_rows.groupby("_normalized_content", sort=False):
        source_ids = sorted(group["id"].astype(str).tolist()) if "id" in group else []
        splits = sorted(group["split"].astype(str).unique().tolist()) if "split" in group else []
        groups.append(
            {
                "duplicate_count": len(group),
                "duplicate_rows_removed": len(group) - 1,
                "source_ids": source_ids,
                "splits": splits,
                "cross_split": len(splits) > 1,
                "preview": content[:160],
            }
        )

    return pd.DataFrame(groups).sort_values(
        ["duplicate_count", "preview"],
        ascending=[False, True],
        ignore_index=True,
    )


def add_text_length(
    dataframe: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Return a copy with the character length of a text column."""
    if source_column not in dataframe:
        raise KeyError(f"Column '{source_column}' is missing")
    result = dataframe.copy()
    result[target_column] = result[source_column].fillna("").astype(str).str.len()
    return result


def conversation_turn_count(record: Mapping[str, Any]) -> int:
    """Estimate the number of turns from common conversation field names."""
    for field in ("utterances", "messages", "turns", "dialog", "conversation"):
        value = record.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)

    for value in record.values():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
    return 0


def add_conversation_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return MSDialog records with turn count and serialized character length."""
    result = dataframe.copy()
    records = result.to_dict(orient="records")
    result["turn_count"] = [conversation_turn_count(record) for record in records]
    result["conversation_char_length"] = [len(str(record)) for record in records]
    return result
