"""IO helpers for indexing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import ChunkDocument


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_chunk_documents(path: Path) -> list[ChunkDocument]:
    """Load chunk records from JSONL as normalized models."""

    return [ChunkDocument.from_row(row) for row in read_jsonl(path)]


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield deterministic fixed-size batches."""

    for start in range(0, len(items), size):
        yield items[start : start + size]
