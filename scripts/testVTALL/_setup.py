
from __future__ import annotations
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from scripts.testVTALL._init import prepare_dataset_and_config, read_csv_dataset

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
config = prepare_dataset_and_config(_CONFIG_PATH)

COLLECTION_SIMPLE = config["qdrant"]["collection_simple"]
COLLECTION_PARENT = config["qdrant"]["collection_parent"]
EMBED_MODEL_NAME = config["model"]["name"]
QDRANT_URL = config["qdrant"]["url"]
PASSAGE_PREFIX = config["model"]["passage_prefix"]
QUERY_PREFIX = config["model"]["query_prefix"]
CONTENT_FIELD = config["datasets"]["content_field"]
REQUIRED_FIELDS: list[str] = config["datasets"]["required_fields"]


def _validate_fields(corpus: list[dict[str, Any]]) -> None:
    """Vérifie que les champs requis par la config sont présents dans le dataset."""
    if not corpus:
        print("Erreur : le dataset est vide.")
        sys.exit(1)
    sample = corpus[0]
    missing = [f for f in REQUIRED_FIELDS if f not in sample]
    if missing:
        print(f"Erreur : champs manquants dans le dataset : {missing}")
        print(f"Champs disponibles : {list(sample.keys())}")
        sys.exit(1)
    if CONTENT_FIELD not in sample:
        print(f"Erreur : CONTENT_FIELD='{CONTENT_FIELD}' introuvable dans le dataset.")
        print(f"Champs disponibles : {list(sample.keys())}")
        sys.exit(1)


def _to_int_id(value: Any, fallback: int) -> int:
    """Convertit un id quelconque en entier déterministe (utile pour les IDs alphanumériques)."""
    if value is None:
        return fallback
    as_str = str(value).strip()
    if not as_str:
        return fallback
    try:
        return int(as_str)
    except ValueError:
        return zlib.crc32(as_str.encode("utf-8"))


def _extract_text_for_indexing(row: dict[str, Any]) -> str:
    """Retourne uniquement le contenu du champ configure dans CONTENT_FIELD."""
    return str(row.get(CONTENT_FIELD) or "").strip()

def get_or_create_all_collection():
    create_index_simple() 
    create_index_parent_doc_quantized()

def create_index_simple():
    """Charge le dataset CSV, indexe dans Qdrant si nécessaire et retourne (client, model, corpus)."""
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    dataset_path = Path(__file__).resolve().parent / config["datasets"]["download_path"]
    corpus = read_csv_dataset(dataset_path)
    _validate_fields(corpus)

    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
    else:
        dim = model.get_embedding_dimension()

    if not client.collection_exists(COLLECTION_SIMPLE):
        client.create_collection(
            COLLECTION_SIMPLE,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field in REQUIRED_FIELDS:
            client.create_payload_index(
                COLLECTION_SIMPLE,
                field,
                PayloadSchemaType.KEYWORD,
            )

    existing_count = client.count(collection_name=COLLECTION_SIMPLE, exact=True).count
    if existing_count > 0:
        print(f"Collection '{COLLECTION_SIMPLE}' deja indexee ({existing_count} points).")
        return client

    texts = [f"{PASSAGE_PREFIX}{_extract_text_for_indexing(c)}" for c in corpus]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    points = [
        PointStruct(id=i, vector=embs[i].tolist(), payload=c)
        for i, c in enumerate(corpus)
    ]
    client.upsert(COLLECTION_SIMPLE, points=points, wait=True)
    print(f"Collection '{COLLECTION_SIMPLE}' indexee avec {len(points)} points.")

    return client


def create_index_parent_doc_quantized() -> None:
    """Indexe une collection parent/child avec un payload base sur REQUIRED_FIELDS."""
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_PARENT):
        client.delete_collection(COLLECTION_PARENT)

    dataset_path = Path(__file__).resolve().parent / config["datasets"]["download_path"]
    articles = read_csv_dataset(dataset_path)
    _validate_fields(articles)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
    else:
        dim = model.get_embedding_dimension()

    client.create_collection(
        COLLECTION_PARENT,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            )
        ),
    )

    # On indexe les champs requis du dataset pour garder la collection coherente
    # avec l'objet source declare dans required_fields.
    for field in REQUIRED_FIELDS:
        client.create_payload_index(COLLECTION_PARENT, field, PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_PARENT, "parent_id", PayloadSchemaType.INTEGER)
    client.create_payload_index(COLLECTION_PARENT, "indexed_at", PayloadSchemaType.DATETIME)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

    points: list[PointStruct] = []
    point_id = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    skipped_articles = 0

    for article_idx, article in enumerate(articles):
        article_text = _extract_text_for_indexing(article)
        if not article_text:
            skipped_articles += 1
            continue

        required_payload = {field: article.get(field) for field in REQUIRED_FIELDS}
        source_id = required_payload.get("id")
        base_id = _to_int_id(source_id, article_idx)

        parents = parent_splitter.split_text(article_text)
        for parent_idx, parent_chunk in enumerate(parents):
            parent_id = base_id * 1000 + parent_idx
            children = child_splitter.split_text(parent_chunk)
            for child_chunk in children:
                point_id += 1
                embedding = model.encode(
                    f"{PASSAGE_PREFIX}{child_chunk}",
                    normalize_embeddings=True,
                )
                payload = {
                    **required_payload,
                    "parent_id": parent_id,
                    "parent_chunk": parent_chunk,
                    "child_chunk": child_chunk,
                    "indexed_at": now_iso,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

    for i in range(0, len(points), 200):
        client.upsert(COLLECTION_PARENT, points=points[i : i + 200], wait=False)

    client.update_collection(COLLECTION_PARENT, optimizer_config={"indexing_threshold": 0})
    print(f"Collection '{COLLECTION_PARENT}' indexee avec {len(points)} chunks enfants.")
    if skipped_articles:
        print(f"Articles ignores faute de contenu exploitable: {skipped_articles}")

    return client

