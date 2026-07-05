"""Qdrant indexing workflow orchestration."""

import logging

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Protocol

from .config import IndexingConfig
from .embeddings import BGEEmbeddingClient
from .io_utils import batched, load_chunk_documents, write_json
from .qdrant_store import (
    build_point,
    collection_stats,
    create_qdrant_client,
    ensure_collection,
    upsert_points,
)


class EmbeddingClient(Protocol):
    """Protocol for embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of input texts."""


def run_indexing_core(
    config: IndexingConfig,
    *,
    embedding_client: EmbeddingClient | None = None,
    qdrant_client: Any | None = None,
) -> dict[str, Any]:
    """Execute indexation without requiring external orchestration."""

    logging.info("Starting indexing workflow with config: %s", asdict(config))
    config.validate()
    chunks = load_chunk_documents(config.chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {config.chunks_path}")

    logging.info("Loaded %d chunks from %s", len(chunks), config.chunks_path)
    if embedding_client is None:
        embedding_client = BGEEmbeddingClient(model=config.embedding_model)

    if qdrant_client is None:
        qdrant_client = create_qdrant_client(config.qdrant_url, api_key=config.qdrant_api_key)

    indexed_count = 0
    embedded_count = 0
    vector_size: int | None = None
    collection_ready = False

    for chunk_batch in batched(chunks, config.embedding_batch_size):
        logging.info(
            "Processing batch of %d chunks (total %d on %d)",
            len(chunk_batch),
            embedded_count,
            len(chunks),
        )
        texts = [chunk.text for chunk in chunk_batch]
        vectors = embedding_client.embed_texts(texts)
        embedded_count += len(vectors)

        if not vectors:
            continue

        if len(vectors) != len(chunk_batch):
            raise RuntimeError(
                "Embedding response size mismatch: "
                f"batch has {len(chunk_batch)} chunks, got {len(vectors)} vectors"
            )

        if vector_size is None:
            vector_size = len(vectors[0])
        if not collection_ready:
            ensure_collection(
                qdrant_client,
                collection_name=config.collection_name,
                vector_size=vector_size,
                recreate_collection=config.recreate_collection,
            )
            collection_ready = True

        points = [build_point(chunk, vector) for chunk, vector in zip(chunk_batch, vectors, strict=True)]
        indexed_count += upsert_points(
            qdrant_client,
            collection_name=config.collection_name,
            points=points,
            batch_size=config.upsert_batch_size,
        )

    logging.info(
        "Indexing workflow completed: %d chunks embedded, %d chunks indexed",
        embedded_count,
        indexed_count,
    )
    stats = collection_stats(qdrant_client, config.collection_name)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            key: str(value) for key, value in asdict(config).items() if key.endswith("_path")
        }
        | {
            key: value for key, value in asdict(config).items() if not key.endswith("_path")
        },
        "counts": {
            "chunks_loaded": len(chunks),
            "chunks_embedded": embedded_count,
            "chunks_indexed": indexed_count,
        },
        "qdrant": {
            "url": config.qdrant_url,
            "collection": config.collection_name,
            "vector_size": vector_size,
        }
        | stats,
    }
    write_json(config.manifest_path, manifest)
    logging.info("Indexing manifest written to %s", config.manifest_path)
    return manifest
