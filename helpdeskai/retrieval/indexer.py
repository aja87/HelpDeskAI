"""Index processed chunks into Qdrant and pgvector."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

import numpy as np

from helpdeskai.retrieval.corpus import ChunkRecord, load_chunks
from helpdeskai.retrieval.embeddings import SentenceTransformerEmbedder
from helpdeskai.retrieval.models import RetrievalConfig


def qdrant_point_id(chunk_id: str) -> str:
    """Return a deterministic UUID acceptable as a Qdrant point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"helpdeskai:{chunk_id}"))


def batched(records: Sequence[ChunkRecord], size: int) -> list[Sequence[ChunkRecord]]:
    """Split records into index batches."""
    return [records[start : start + size] for start in range(0, len(records), size)]


def create_qdrant_collection(
    client,
    config: RetrievalConfig,
    vector_size: int,
    *,
    recreate: bool,
) -> None:
    """Create Qdrant collection and payload indexes."""
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    if client.collection_exists(config.collection_name):
        if not recreate:
            return
        client.delete_collection(config.collection_name)
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    for field in ("document_id", "product", "versions", "date", "category", "tenant"):
        client.create_payload_index(
            collection_name=config.collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def upsert_qdrant(
    client,
    records: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    config: RetrievalConfig,
) -> None:
    """Upsert one batch of chunks into Qdrant."""
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=qdrant_point_id(record.chunk_id),
            vector=embeddings[index].tolist(),
            payload=record.payload,
        )
        for index, record in enumerate(records)
    ]
    client.upsert(collection_name=config.collection_name, points=points, wait=True)


def setup_pgvector(conn, config: RetrievalConfig, vector_size: int, *, recreate: bool) -> None:
    """Create the pgvector comparison table."""
    from pgvector.psycopg import register_vector

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    with conn.cursor() as cur:
        if recreate:
            cur.execute(f"DROP TABLE IF EXISTS {config.pgvector_table}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.pgvector_table} (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({vector_size}) NOT NULL,
                product TEXT,
                versions TEXT[],
                published_at DATE,
                category TEXT,
                tenant TEXT,
                payload JSONB NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {config.pgvector_table}_embedding_hnsw
            ON {config.pgvector_table}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {config.pgvector_table}_metadata_idx "
            f"ON {config.pgvector_table} (product, category, tenant)"
        )


def upsert_pgvector(
    conn,
    records: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    config: RetrievalConfig,
) -> None:
    """Mirror one batch of chunks into PostgreSQL/pgvector."""
    from psycopg.types.json import Jsonb

    with conn.cursor() as cur:
        for index, record in enumerate(records):
            metadata = record.metadata
            cur.execute(
                f"""
                INSERT INTO {config.pgvector_table}
                (chunk_id, document_id, content, embedding, product, versions, published_at,
                 category, tenant, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    product = EXCLUDED.product,
                    versions = EXCLUDED.versions,
                    published_at = EXCLUDED.published_at,
                    category = EXCLUDED.category,
                    tenant = EXCLUDED.tenant,
                    payload = EXCLUDED.payload
                """,
                (
                    record.chunk_id,
                    record.document_id,
                    record.content,
                    embeddings[index],
                    metadata.get("product"),
                    metadata.get("versions") or [],
                    metadata.get("date"),
                    metadata.get("category"),
                    metadata.get("tenant"),
                    Jsonb(record.payload),
                ),
            )


def index_corpus(
    *,
    config: RetrievalConfig = RetrievalConfig(),
    recreate: bool = True,
    index_qdrant: bool = True,
    index_pgvector: bool = True,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Index the processed corpus into the configured retrieval stores."""
    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report(f"Loading chunks from {config.corpus_path}")
    records = load_chunks(config.corpus_path)
    report(f"Loaded {len(records)} chunks")

    report(f"Loading embedding model {config.model_name}")
    embedder = SentenceTransformerEmbedder(config.model_name)
    report(f"Embedding dimension: {embedder.dimension}")

    qdrant_client = None
    if index_qdrant:
        from qdrant_client import QdrantClient

        report(f"Connecting to Qdrant at {config.qdrant_url}")
        qdrant_client = QdrantClient(url=config.qdrant_url)
        action = "Recreating" if recreate else "Using"
        report(f"{action} Qdrant collection {config.collection_name}")
        create_qdrant_collection(qdrant_client, config, embedder.dimension, recreate=recreate)
    else:
        report("Skipping Qdrant indexing")

    pg_conn = None
    if index_pgvector:
        import psycopg

        report(f"Connecting to pgvector at {config.pgvector_dsn}")
        pg_conn = psycopg.connect(config.pgvector_dsn, autocommit=True)
        action = "Recreating" if recreate else "Using"
        report(f"{action} pgvector table {config.pgvector_table}")
        setup_pgvector(pg_conn, config, embedder.dimension, recreate=recreate)
    else:
        report("Skipping pgvector indexing")

    try:
        batches = batched(records, config.batch_size)
        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            report(
                f"Embedding batch {batch_index}/{total_batches} "
                f"({len(batch)} chunks)"
            )
            embeddings = embedder.encode_documents(
                [record.content for record in batch],
                batch_size=config.batch_size,
            )
            if qdrant_client is not None:
                report(f"Upserting batch {batch_index}/{total_batches} to Qdrant")
                upsert_qdrant(qdrant_client, batch, embeddings, config)
            if pg_conn is not None:
                report(f"Upserting batch {batch_index}/{total_batches} to pgvector")
                upsert_pgvector(pg_conn, batch, embeddings, config)
    finally:
        if pg_conn is not None:
            pg_conn.close()
            report("Closed pgvector connection")

    report(f"Indexed {len(records)} chunks")
    return len(records)
