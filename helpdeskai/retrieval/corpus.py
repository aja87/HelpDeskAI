"""Corpus loading and metadata normalization for retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from helpdeskai.ingestion.io import read_jsonl
from helpdeskai.retrieval.models import SearchFilters

TENANTS = ("novacloud-core", "novacloud-identity", "novacloud-ops")


@dataclass(frozen=True)
class ChunkRecord:
    """Index-ready chunk with flattened retrieval metadata."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    payload: dict[str, Any]


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _stable_tenant(document_id: str) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return TENANTS[digest[0] % len(TENANTS)]


def _normalize_versions(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def flatten_chunk(record: Mapping[str, Any]) -> ChunkRecord:
    """Flatten one ingestion chunk into a retrieval payload."""
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    chunk_id = str(record["chunk_id"])
    document_id = str(record["document_id"])
    content = str(record["content"])
    product = _metadata_value(metadata, "product")
    versions = _normalize_versions(_metadata_value(metadata, "versions"))
    published_at = _metadata_value(metadata, "date")
    category = _metadata_value(metadata, "category")
    title = metadata.get("title") or record.get("title")
    tenant = _metadata_value(metadata, "tenant") or _stable_tenant(document_id)

    flattened = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": content,
        "content_hash": record.get("content_hash"),
        "position": record.get("position"),
        "product": str(product) if product is not None else None,
        "versions": versions,
        "date": str(published_at) if published_at is not None else None,
        "category": str(category) if category is not None else None,
        "title": str(title) if title is not None else None,
        "tenant": str(tenant),
        "source_ids": list(record.get("source_ids") or []),
        "splits": list(record.get("splits") or []),
        "chunking": record.get("chunking"),
        "raw_metadata": dict(metadata),
    }
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        metadata=flattened,
        payload=flattened,
    )


def load_chunks(path: Path) -> list[ChunkRecord]:
    """Load processed TechQA chunks from JSONL."""
    return [flatten_chunk(record) for record in read_jsonl(path)]


def _valid_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def matches_filters(record: ChunkRecord | Mapping[str, Any], filters: SearchFilters | None) -> bool:
    """Return whether a chunk or payload satisfies metadata filters."""
    if filters is None:
        return True
    metadata = record.metadata if isinstance(record, ChunkRecord) else record
    if filters.product and metadata.get("product") != filters.product:
        return False
    if filters.version and filters.version not in set(metadata.get("versions") or []):
        return False
    if filters.tenant and metadata.get("tenant") != filters.tenant:
        return False
    record_date = _valid_iso(metadata.get("date"))
    if filters.date_from:
        lower = _valid_iso(filters.date_from)
        if lower and (record_date is None or record_date < lower):
            return False
    if filters.date_to:
        upper = _valid_iso(filters.date_to)
        if upper and (record_date is None or record_date > upper):
            return False
    return True


def filter_chunks(
    records: Sequence[ChunkRecord],
    filters: SearchFilters | None,
) -> list[ChunkRecord]:
    """Filter records with the same semantics used by sparse retrieval."""
    return [record for record in records if matches_filters(record, filters)]
