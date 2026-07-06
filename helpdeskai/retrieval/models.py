"""Shared retrieval models and configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SearchMode(StrEnum):
    """Supported retrieval modes."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters applied consistently across retrieval modes."""

    product: str | None = None
    version: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    tenant: str | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    """Runtime configuration for retrieval services."""

    collection_name: str = field(
        default_factory=lambda: os.environ.get(
            "HELPDESKAI_QDRANT_COLLECTION",
            "helpdeskai_techqa_chunks",
        )
    )
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    pgvector_dsn: str = field(
        default_factory=lambda: os.environ.get(
            "PGVECTOR_DSN",
            "postgresql://postgres:postgres@localhost:5433/helpdeskai",
        )
    )
    pgvector_table: str = "retrieval_chunks"
    model_name: str = field(
        default_factory=lambda: os.environ.get("HELPDESKAI_EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    corpus_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HELPDESKAI_CORPUS_PATH", "data/processed/techqa/chunks.jsonl")
        )
    )
    batch_size: int = 64
    rrf_k: int = 60


@dataclass(frozen=True)
class SearchResult:
    """A normalized retrieval result returned by the public API."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    mode: SearchMode
    metadata: dict[str, Any] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
