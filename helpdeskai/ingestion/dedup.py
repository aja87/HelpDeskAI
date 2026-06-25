"""Canonical document deduplication stage."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


def content_hash(text: str) -> str:
    """Return the stable hash used for canonical deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_documents(
    payloads: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate normalized documents and preserve source aliases."""
    canonical_by_hash: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        digest = content_hash(payload["text"])
        document_id = f"techqa-{digest[:16]}"
        canonical = canonical_by_hash.get(digest)
        if canonical is None:
            canonical = {
                "document_id": document_id,
                "content_hash": digest,
                "text": payload["text"],
                "source_ids": [],
                "splits": [],
                "source_record_count": 0,
                "normalization_version": payload["normalization_version"],
                "extraction_methods": [],
                "source_types": [],
            }
            canonical_by_hash[digest] = canonical

        canonical["source_ids"].append(payload["source_id"])
        if payload["split"] not in canonical["splits"]:
            canonical["splits"].append(payload["split"])
        if payload["extraction_method"] not in canonical["extraction_methods"]:
            canonical["extraction_methods"].append(payload["extraction_method"])
        if payload["source_type"] not in canonical["source_types"]:
            canonical["source_types"].append(payload["source_type"])
        canonical["source_record_count"] += 1

    documents = sorted(canonical_by_hash.values(), key=lambda item: item["document_id"])
    for document in documents:
        document["source_ids"].sort()
        document["splits"].sort()
        document["extraction_methods"].sort()
        document["source_types"].sort()
    return documents


def deduplicate_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate chunk content while preserving first occurrence."""
    seen: set[str] = set()
    unique = []
    for chunk in chunks:
        digest = chunk.get("content_hash") or content_hash(chunk.get("content", ""))
        if digest in seen:
            continue
        seen.add(digest)
        item = dict(chunk)
        item["content_hash"] = digest
        unique.append(item)
    return unique
