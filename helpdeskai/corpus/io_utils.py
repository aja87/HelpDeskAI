"""File IO helpers for corpus download outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import CHECKSUM_FILE


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records to JSONL with one object per line."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def sha256(path: Path) -> str:
    """Compute the SHA-256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(base_dir: Path) -> bool:
    """Validate all expected artifact hashes from checksum manifest."""

    checksum_path = base_dir / CHECKSUM_FILE
    if not checksum_path.exists():
        return False
    expected = json.loads(checksum_path.read_text(encoding="utf-8"))
    for filename, expected_hash in expected.items():
        file_path = base_dir / filename
        if not file_path.exists():
            return False
        if file_path.stat().st_size == 0:
            return False
        if sha256(file_path) != expected_hash:
            return False
    return True
