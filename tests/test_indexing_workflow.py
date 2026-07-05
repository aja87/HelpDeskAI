from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helpdeskai.indexing.config import IndexingConfig
from helpdeskai.indexing.io_utils import batched
from helpdeskai.indexing.models import ChunkDocument
from helpdeskai.indexing.workflow import run_indexing_core


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self._exists = False
        self.create_calls = 0
        self.delete_calls = 0
        self.upserts: list[list[Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self._exists

    def delete_collection(self, collection_name: str) -> None:
        self.delete_calls += 1
        self._exists = False

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.create_calls += 1
        self._exists = True

    def upsert(self, collection_name: str, points: list[Any], wait: bool) -> None:
        self.upserts.append(points)

    def get_collection(self, collection_name: str) -> Any:
        total = sum(len(batch) for batch in self.upserts)
        return type(
            "CollectionInfo",
            (),
            {"status": "green", "points_count": total, "vectors_count": total},
        )()


def _write_chunks(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([json.dumps(row) for row in rows]) + "\n", encoding="utf-8")


def test_batched_splits_input_deterministically() -> None:
    chunks = list(batched([1, 2, 3, 4, 5], 2))
    assert chunks == [[1, 2], [3, 4], [5]]


def test_chunk_document_requires_chunk_id_and_text() -> None:
    row = {
        "chunk_id": "C1",
        "doc_id": "D1",
        "chunk_index": 0,
        "text": "hello",
        "source": "techqa",
        "title": "t",
        "product": "p",
        "version": "v",
        "category": "c",
        "date": "",
        "strategy": "recursive",
    }
    doc = ChunkDocument.from_row(row)
    assert doc.chunk_id == "C1"
    assert doc.payload()["text"] == "hello"


def test_run_indexing_core_indexes_all_chunks(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "reports" / "indexing_manifest.json"
    rows = [
        {
            "chunk_id": "C1",
            "doc_id": "D1",
            "chunk_index": 0,
            "text": "first",
            "source": "techqa",
            "title": "t1",
            "product": "p1",
            "version": "1",
            "category": "troubleshooting",
            "date": "",
            "strategy": "recursive",
        },
        {
            "chunk_id": "C2",
            "doc_id": "D2",
            "chunk_index": 1,
            "text": "second",
            "source": "techqa",
            "title": "t2",
            "product": "p2",
            "version": "2",
            "category": "faq",
            "date": "",
            "strategy": "recursive",
        },
    ]
    _write_chunks(chunks_path, rows)

    config = IndexingConfig(
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        embedding_batch_size=1,
        upsert_batch_size=1,
        recreate_collection=True,
    )
    qdrant = FakeQdrantClient()

    result = run_indexing_core(
        config,
        embedding_client=FakeEmbeddingClient(),
        qdrant_client=qdrant,
    )

    assert result["counts"]["chunks_loaded"] == 2
    assert result["counts"]["chunks_indexed"] == 2
    assert qdrant.create_calls == 1
    assert len(qdrant.upserts) == 2
    assert manifest_path.exists()
