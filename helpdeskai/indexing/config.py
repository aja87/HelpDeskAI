"""Configuration for phase-3 Qdrant indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
INDEXING_REPORTS_DIR = Path("reports/indexing")
DEFAULT_CHUNKS_PATH = PROCESSED_DIR / "techqa_chunks.jsonl"
DEFAULT_MANIFEST_PATH = INDEXING_REPORTS_DIR / "indexing_manifest.json"
LOG_FILE = "indexing.log"

@dataclass(slots=True)
class IndexingConfig:
    """Runtime configuration for Qdrant indexation."""

    chunks_path: Path = DEFAULT_CHUNKS_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "helpdeskai-techqa"
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 64
    upsert_batch_size: int = 128
    recreate_collection: bool = True

    def validate(self) -> None:
        """Validate configuration values before execution."""

        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be > 0")
        if self.upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be > 0")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
