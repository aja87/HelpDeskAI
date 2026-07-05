"""Data structures used during indexation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChunkDocument:
    """Normalized chunk row ready for embedding and indexing."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source: str
    title: str
    product: str
    version: str
    category: str
    date: str
    strategy: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ChunkDocument":
        """Build a chunk document from a JSONL row."""

        chunk_id = str(row.get("chunk_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not chunk_id:
            raise ValueError("chunk_id is required")
        if not text:
            raise ValueError(f"text is required for chunk_id={chunk_id}")

        return cls(
            chunk_id=chunk_id,
            doc_id=str(row.get("doc_id", "")),
            chunk_index=int(row.get("chunk_index", 0)),
            text=text,
            source=str(row.get("source", "")),
            title=str(row.get("title", "")),
            product=str(row.get("product", "")),
            version=str(row.get("version", "")),
            category=str(row.get("category", "")),
            date=str(row.get("date", "")),
            strategy=str(row.get("strategy", "")),
        )

    def payload(self) -> dict[str, Any]:
        """Build the Qdrant payload for this chunk."""

        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "product": self.product,
            "version": self.version,
            "category": self.category,
            "date": self.date,
            "strategy": self.strategy,
        }
