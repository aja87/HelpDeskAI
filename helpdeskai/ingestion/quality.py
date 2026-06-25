"""Quality gates and Evidently report generation for index-ready chunks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


class CorpusQualityError(RuntimeError):
    """Raised when index-ready chunks violate the corpus contract."""


def validate_chunks(chunks: Sequence[dict], document_ids: set[str]) -> dict:
    """Validate blocking corpus invariants and return quality statistics."""
    if not chunks:
        raise CorpusQualityError("chunk corpus is empty")
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    duplicates = [value for value, count in Counter(chunk_ids).items() if count > 1]
    if duplicates:
        raise CorpusQualityError(f"duplicate chunk IDs: {duplicates[:5]}")

    content_hashes = []
    for index, chunk in enumerate(chunks):
        if not str(chunk.get("content", "")).strip():
            raise CorpusQualityError(f"chunk {index} has empty content")
        if chunk.get("document_id") not in document_ids:
            raise CorpusQualityError(f"chunk {index} references an unknown document")
        token_count = chunk.get("token_count")
        if not isinstance(token_count, int) or not 1 <= token_count <= 512:
            raise CorpusQualityError(f"chunk {index} has invalid token_count={token_count}")
        if not isinstance(chunk.get("metadata"), dict):
            raise CorpusQualityError(f"chunk {index} has invalid metadata")
        content_hashes.append(chunk["content_hash"])

    metadata_fields = ("category", "product", "versions", "date")
    return {
        "chunks": len(chunks),
        "unique_chunk_ids": len(set(chunk_ids)),
        "exact_duplicate_content": sum(count - 1 for count in Counter(content_hashes).values()),
        "metadata_coverage": {
            field: sum(bool(chunk["metadata"].get(field, {}).get("value")) for chunk in chunks)
            for field in metadata_fields
        },
        "min_tokens": min(chunk["token_count"] for chunk in chunks),
        "max_tokens": max(chunk["token_count"] for chunk in chunks),
    }


def _quality_dataframe(chunks: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "token_count": chunk["token_count"],
                "content_length": len(chunk["content"]),
                "category_present": bool(metadata["category"]["value"]),
                "product_present": bool(metadata["product"]["value"]),
                "versions_present": bool(metadata["versions"]["value"]),
                "date_present": bool(metadata["date"]["value"]),
                "content_hash": chunk["content_hash"],
            }
        )
    return pd.DataFrame(rows)


def generate_evidently_report(chunks: Sequence[dict], output_path: Path) -> Path:
    """Generate an Evidently HTML data-summary report."""
    from evidently.metric_preset import DataQualityPreset
    from evidently.report import Report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = _quality_dataframe(chunks)
    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=None, current_data=dataframe)
    report.save_html(str(output_path))
    return output_path
