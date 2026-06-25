"""Atomic persistence stage for normalized TechQA outputs."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from helpdeskai.ingestion.enrich import metadata_coverage
from helpdeskai.ingestion.exceptions import TechQAIngestionError
from helpdeskai.ingestion.io import write_jsonl
from helpdeskai.ingestion.normalize import NORMALIZATION_VERSION


def build_ingestion_manifest(
    documents: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    *,
    source_document_count: int,
    html_source_count: int,
    documents_path: Path,
) -> dict[str, Any]:
    """Build the ingestion manifest from stage outputs."""
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "input": {"documents": documents_path.as_posix()},
        "normalization_version": NORMALIZATION_VERSION,
        "source_documents": source_document_count,
        "canonical_documents": len(documents),
        "removed_duplicate_rows": source_document_count - len(documents),
        "duplicate_groups": sum(
            document["source_record_count"] > 1 for document in documents
        ),
        "chunks": len(chunks),
        "html_like_sources": html_source_count,
        "metadata_coverage": metadata_coverage(documents),
        "chunking": chunks[0]["chunking"] if chunks else None,
    }


def persist_ingestion(
    output_dir: Path,
    documents: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    force: bool,
) -> Path:
    """Atomically publish normalized documents, chunks, and manifest."""
    expected = tuple(
        output_dir / filename
        for filename in (
            "documents.jsonl",
            "chunks.jsonl",
            "manifest.json",
        )
    )
    if any(path.exists() for path in expected) and not force:
        raise TechQAIngestionError(
            "Processed output already exists. Use --force to replace it"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".techqa-ingestion-",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary)
        write_jsonl(staging / "documents.jsonl", documents)
        write_jsonl(staging / "chunks.jsonl", chunks)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "documents.jsonl",
            "chunks.jsonl",
            "manifest.json",
        ):
            destination = output_dir / filename
            if destination.exists():
                destination.unlink()
            (staging / filename).replace(destination)
    return output_dir / "manifest.json"
