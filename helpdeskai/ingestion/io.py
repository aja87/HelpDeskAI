"""JSONL input/output helpers for ingestion and corpus scripts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from helpdeskai.ingestion.exceptions import TechQAIngestionError


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file."""
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TechQAIngestionError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise TechQAIngestionError(f"Unable to read {path}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write JSON objects to JSONL and return the record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            count += 1
    return count
