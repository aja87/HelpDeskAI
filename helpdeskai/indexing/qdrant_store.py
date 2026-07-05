"""Qdrant collection management and upsert utilities."""

from __future__ import annotations

import logging
import uuid

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .io_utils import batched
from .models import ChunkDocument


def create_qdrant_client(url: str, api_key: str | None = None) -> QdrantClient:
    """Create a Qdrant client configured for local or remote usage."""

    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    recreate_collection: bool,
) -> None:
    """Create or recreate collection with cosine distance."""

    collection_exists = client.collection_exists(collection_name=collection_name)
    if collection_exists and recreate_collection:
        client.delete_collection(collection_name=collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def to_qdrant_id(chunk_id: str) -> str:
    # deterministic UUID: same chunk_id -> same point id across reindex runs
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"helpdeskai:{chunk_id}"))

def build_point(chunk: ChunkDocument, vector: list[float]) -> models.PointStruct:
    return models.PointStruct(
        id=to_qdrant_id(chunk.chunk_id),
        vector=vector,
        payload=chunk.payload(),
    )


def upsert_points(
    client: QdrantClient,
    *,
    collection_name: str,
    points: list[models.PointStruct],
    batch_size: int,
) -> int:
    """Upsert points in deterministic batches."""

    inserted = 0
    for chunk in batched(points, batch_size):
        logging.info("Upserting batch of %d points into collection %s", len(chunk), collection_name)
        client.upsert(collection_name=collection_name, points=chunk, wait=True)
        inserted += len(chunk)
    return inserted


def collection_stats(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    """Return lightweight stats from a collection info object."""

    info = client.get_collection(collection_name=collection_name)
    return {
        "status": str(info.status),
        "points_count": int(info.points_count or 0),
        "vectors_count": int(info.vectors_count or 0),
    }
