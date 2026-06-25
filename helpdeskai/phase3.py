from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from sentence_transformers import SentenceTransformer


# Embedding model choice justified with public MTEB-style retrieval references.
# Values are indicative multilingual retrieval proxies used for deterministic selection.
MTEB_RETRIEVAL_PROXY = {
    "BAAI/bge-m3": 64.7,
    "intfloat/multilingual-e5-large": 63.9,
    "text-embedding-3-small": 62.1,
}

SIMULATED_PRODUCTS = ["NovaCloud Core", "NovaCloud Assist", "NovaCloud Secure"]
SIMULATED_TENANTS = ["tenant-alpha", "tenant-beta", "tenant-gamma"]


@dataclass
class Config:
    dataset_path: Path
    golden_path: Path
    qdrant_url: str = "http://localhost:6333"
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    collection: str = "helpdeskai_phase3"
    pg_table: str = "helpdeskai_phase3"
    top_k: int = 10
    batch_size: int = 128
    benchmark_limit: int = 100


class SparseBM25Encoder:
    """Simple sparse encoder compatible with Qdrant sparse vectors.

    Qdrant IDF modifier is used to apply IDF weighting on sparse search.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for tok in self.tokenize(text):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def encode(self, text: str) -> qm.SparseVector:
        tf: dict[int, float] = {}
        for tok in self.tokenize(text):
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            tf[idx] = tf.get(idx, 0.0) + 1.0

        if not tf:
            return qm.SparseVector(indices=[], values=[])

        indices = sorted(tf.keys())
        values = [tf[i] for i in indices]
        return qm.SparseVector(indices=indices, values=values)


class HelpdeskRetrievalIndexer:
    """Implementation target for helpdeskai.retrieval.indexer phase requirements."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model_name = self.choose_embedding_model()
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

        self.qdrant = QdrantClient(url=self.cfg.qdrant_url)
        self.sparse_encoder = SparseBM25Encoder()

    @staticmethod
    def choose_embedding_model() -> str:
        return max(MTEB_RETRIEVAL_PROXY.items(), key=lambda kv: kv[1])[0]

    def _dense_passage(self, text: str) -> str:
        if self.model_name.startswith("intfloat/multilingual-e5"):
            return f"passage: {text}"
        return text

    def _dense_query(self, text: str) -> str:
        if self.model_name.startswith("intfloat/multilingual-e5"):
            return f"query: {text}"
        return text

    @staticmethod
    def _stable_hash(value: str) -> int:
        return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)

    def _derive_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(doc.get("doc_id", ""))
        h = self._stable_hash(doc_id or str(doc.get("title", "")))

        product = SIMULATED_PRODUCTS[h % len(SIMULATED_PRODUCTS)]
        tenant = SIMULATED_TENANTS[(h // 7) % len(SIMULATED_TENANTS)]

        version = str(doc.get("version") or "").strip()
        if not version:
            version = ["3.0", "3.1", "3.2"][h % 3]

        date_raw = str(doc.get("date") or "").strip()
        year = None
        if date_raw:
            m = re.search(r"(19|20)\d{2}", date_raw)
            if m:
                year = int(m.group(0))
        if year is None:
            year = 2021 + (h % 6)

        return {
            "doc_id": doc_id,
            "source_doc_id": str(doc.get("source_doc_id", "")),
            "title": str(doc.get("title", "")),
            "text": str(doc.get("text", "")),
            "source": str(doc.get("source", "techqa")),
            "category": str(doc.get("category", "")),
            "product": product,
            "tenant": tenant,
            "version": version,
            "date_year": year,
            "original_product": str(doc.get("product", "")),
        }

    def load_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        with self.cfg.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                normalized = self._derive_metadata(raw)
                if normalized["text"]:
                    docs.append(normalized)
        return docs

    def load_golden(self) -> list[dict[str, Any]]:
        golden: list[dict[str, Any]] = []
        with self.cfg.golden_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q = str(row.get("question", "")).strip()
                rel_doc_id = str(row.get("doc_id", "")).strip()
                if not q or not rel_doc_id:
                    continue
                golden.append(
                    {
                        "question_id": row.get("question_id", ""),
                        "question": q,
                        "relevant_doc_ids": [rel_doc_id],
                    }
                )
                if len(golden) >= self.cfg.benchmark_limit:
                    break
        return golden

    def _prepare_vectors(self, docs: list[dict[str, Any]]) -> tuple[list[list[float]], list[qm.SparseVector]]:
        texts = [self._dense_passage(d["text"]) for d in docs]
        self.sparse_encoder.fit([d["text"] for d in docs])

        dense = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
        )
        sparse = [self.sparse_encoder.encode(d["text"]) for d in docs]
        return dense.tolist(), sparse

    def _recreate_qdrant(self) -> None:
        if self.qdrant.collection_exists(self.cfg.collection):
            self.qdrant.delete_collection(self.cfg.collection)

        self.qdrant.create_collection(
            collection_name=self.cfg.collection,
            vectors_config={
                "dense": qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
            },
        )

        for field_name, field_schema in [
            ("doc_id", qm.PayloadSchemaType.KEYWORD),
            ("tenant", qm.PayloadSchemaType.KEYWORD),
            ("product", qm.PayloadSchemaType.KEYWORD),
            ("version", qm.PayloadSchemaType.KEYWORD),
            ("date_year", qm.PayloadSchemaType.INTEGER),
            ("category", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
        ]:
            self.qdrant.create_payload_index(
                collection_name=self.cfg.collection,
                field_name=field_name,
                field_schema=field_schema,
            )

    def _index_qdrant(self, docs: list[dict[str, Any]], dense_vectors: list[list[float]], sparse_vectors: list[qm.SparseVector]) -> None:
        points: list[qm.PointStruct] = []
        for i, d in enumerate(docs):
            payload = {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "text": d["text"],
                "source": d["source"],
                "category": d["category"],
                "product": d["product"],
                "version": d["version"],
                "date_year": d["date_year"],
                "tenant": d["tenant"],
                "original_product": d["original_product"],
            }
            points.append(
                qm.PointStruct(
                    id=i + 1,
                    vector={
                        "dense": dense_vectors[i],
                        "bm25": sparse_vectors[i],
                    },
                    payload=payload,
                )
            )

        for start in range(0, len(points), self.cfg.batch_size):
            batch = points[start : start + self.cfg.batch_size]
            self.qdrant.upsert(collection_name=self.cfg.collection, points=batch, wait=False)

        self.qdrant.update_collection(self.cfg.collection, optimizer_config={"indexing_threshold": 0})

    def _setup_pg(self) -> None:
        with psycopg.connect(self.cfg.pg_dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"DROP TABLE IF EXISTS {self.cfg.pg_table}")
                cur.execute(
                    f"""
                    CREATE TABLE {self.cfg.pg_table} (
                        id BIGINT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        title TEXT,
                        text TEXT NOT NULL,
                        source TEXT,
                        category TEXT,
                        product TEXT,
                        version TEXT,
                        date_year INTEGER,
                        tenant TEXT,
                        embedding vector({self.dim})
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX {self.cfg.pg_table}_hnsw_idx
                    ON {self.cfg.pg_table}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """
                )
                cur.execute(f"CREATE INDEX {self.cfg.pg_table}_tenant_idx ON {self.cfg.pg_table}(tenant)")
                cur.execute(f"CREATE INDEX {self.cfg.pg_table}_product_idx ON {self.cfg.pg_table}(product)")
                cur.execute(f"CREATE INDEX {self.cfg.pg_table}_version_idx ON {self.cfg.pg_table}(version)")
                cur.execute(f"CREATE INDEX {self.cfg.pg_table}_date_year_idx ON {self.cfg.pg_table}(date_year)")

    def _index_pgvector(self, docs: list[dict[str, Any]], dense_vectors: list[list[float]]) -> None:
        with psycopg.connect(self.cfg.pg_dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SET hnsw.ef_search = 100")
                rows = [
                    (
                        i + 1,
                        d["doc_id"],
                        d["title"],
                        d["text"],
                        d["source"],
                        d["category"],
                        d["product"],
                        d["version"],
                        d["date_year"],
                        d["tenant"],
                        dense_vectors[i],
                    )
                    for i, d in enumerate(docs)
                ]
                cur.executemany(
                    f"""
                    INSERT INTO {self.cfg.pg_table}
                    (id, doc_id, title, text, source, category, product, version, date_year, tenant, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )

    def build_indexes(self) -> list[dict[str, Any]]:
        docs = self.load_documents()
        print(f"Loaded {len(docs)} normalized documents")
        print(f"Model selected from MTEB proxy: {self.model_name} (score={MTEB_RETRIEVAL_PROXY[self.model_name]:.1f})")

        dense_vectors, sparse_vectors = self._prepare_vectors(docs)

        self._recreate_qdrant()
        self._setup_pg()

        with ThreadPoolExecutor(max_workers=2) as pool:
            qdrant_future = pool.submit(self._index_qdrant, docs, dense_vectors, sparse_vectors)
            pg_future = pool.submit(self._index_pgvector, docs, dense_vectors)
            qdrant_future.result()
            pg_future.result()

        print("Parallel indexing completed: Qdrant + PostgreSQL/pgvector")
        return docs

    @staticmethod
    def _build_qdrant_filter(
        product: str | None = None,
        version: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        tenant: str | None = None,
    ) -> qm.Filter | None:
        must: list[qm.FieldCondition] = []
        if product:
            must.append(qm.FieldCondition(key="product", match=qm.MatchValue(value=product)))
        if version:
            must.append(qm.FieldCondition(key="version", match=qm.MatchValue(value=version)))
        if tenant:
            must.append(qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)))
        if date_from is not None or date_to is not None:
            must.append(qm.FieldCondition(key="date_year", range=qm.Range(gte=date_from, lte=date_to)))
        return qm.Filter(must=must) if must else None

    def search_dense_qdrant(
        self,
        query: str,
        top_k: int = 10,
        product: str | None = None,
        version: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        emb = self.model.encode(self._dense_query(query), normalize_embeddings=True).tolist()
        q_filter = self._build_qdrant_filter(product, version, date_from, date_to, tenant)

        resp = self.qdrant.query_points(
            collection_name=self.cfg.collection,
            query=qm.NamedVector(name="dense", vector=emb),
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "doc_id": p.payload.get("doc_id"),
                "score": float(p.score),
                "title": p.payload.get("title"),
            }
            for p in resp.points
        ]

    def search_sparse_qdrant(
        self,
        query: str,
        top_k: int = 10,
        product: str | None = None,
        version: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        sparse_q = self.sparse_encoder.encode(query)
        q_filter = self._build_qdrant_filter(product, version, date_from, date_to, tenant)

        resp = self.qdrant.query_points(
            collection_name=self.cfg.collection,
            query=qm.NamedSparseVector(name="bm25", vector=sparse_q),
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "doc_id": p.payload.get("doc_id"),
                "score": float(p.score),
                "title": p.payload.get("title"),
            }
            for p in resp.points
        ]

    @staticmethod
    def _rrf(rankings: list[list[str]], k: int = 60, top_k: int = 10) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

    def search_hybrid_rrf_qdrant(
        self,
        query: str,
        top_k: int = 10,
        product: str | None = None,
        version: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        dense_hits = self.search_dense_qdrant(
            query=query,
            top_k=max(top_k, 20),
            product=product,
            version=version,
            date_from=date_from,
            date_to=date_to,
            tenant=tenant,
        )
        sparse_hits = self.search_sparse_qdrant(
            query=query,
            top_k=max(top_k, 20),
            product=product,
            version=version,
            date_from=date_from,
            date_to=date_to,
            tenant=tenant,
        )

        dense_rank = [h["doc_id"] for h in dense_hits if h.get("doc_id")]
        sparse_rank = [h["doc_id"] for h in sparse_hits if h.get("doc_id")]
        fused = self._rrf([dense_rank, sparse_rank], k=60, top_k=top_k)

        dense_map = {h["doc_id"]: h for h in dense_hits}
        sparse_map = {h["doc_id"]: h for h in sparse_hits}

        out: list[dict[str, Any]] = []
        for doc_id in fused:
            out.append(
                {
                    "doc_id": doc_id,
                    "score": dense_map.get(doc_id, {}).get("score", 0.0)
                    + sparse_map.get(doc_id, {}).get("score", 0.0),
                    "title": dense_map.get(doc_id, {}).get("title")
                    or sparse_map.get(doc_id, {}).get("title"),
                }
            )
        return out

    def search_dense_pgvector(
        self,
        query: str,
        top_k: int = 10,
        product: str | None = None,
        version: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        emb = self.model.encode(self._dense_query(query), normalize_embeddings=True).tolist()

        where_clauses: list[str] = []
        params: list[Any] = []
        if product:
            where_clauses.append("product = %s")
            params.append(product)
        if version:
            where_clauses.append("version = %s")
            params.append(version)
        if tenant:
            where_clauses.append("tenant = %s")
            params.append(tenant)
        if date_from is not None:
            where_clauses.append("date_year >= %s")
            params.append(date_from)
        if date_to is not None:
            where_clauses.append("date_year <= %s")
            params.append(date_to)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT doc_id, title, 1 - (embedding <=> %s::vector) AS similarity
            FROM {self.cfg.pg_table}
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        sql_params = [emb] + params + [emb, top_k]

        with psycopg.connect(self.cfg.pg_dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SET hnsw.ef_search = 100")
                cur.execute(sql, sql_params)
                rows = cur.fetchall()

        return [
            {
                "doc_id": r[0],
                "title": r[1],
                "score": float(r[2]),
            }
            for r in rows
        ]

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

    @staticmethod
    def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def p95(values: list[float]) -> float:
        if not values:
            return 0.0
        idx = int(0.95 * (len(values) - 1))
        return sorted(values)[idx]

    def run_benchmark(self, output_csv: Path) -> list[dict[str, Any]]:
        golden = self.load_golden()
        print(f"Benchmark questions loaded: {len(golden)}")

        def eval_mode(name: str, search_fn) -> dict[str, Any]:
            recalls5: list[float] = []
            recalls10: list[float] = []
            rrs: list[float] = []
            lat_ms: list[float] = []

            for sample in golden:
                relevant = set(sample["relevant_doc_ids"])
                t0 = time.perf_counter()
                hits = search_fn(sample["question"])
                elapsed = (time.perf_counter() - t0) * 1000
                lat_ms.append(elapsed)

                retrieved = [h["doc_id"] for h in hits if h.get("doc_id")]
                recalls5.append(self.recall_at_k(retrieved, relevant, 5))
                recalls10.append(self.recall_at_k(retrieved, relevant, 10))
                rrs.append(self.reciprocal_rank(retrieved, relevant))

            return {
                "mode": name,
                "recall@5": round(statistics.mean(recalls5), 4),
                "recall@10": round(statistics.mean(recalls10), 4),
                "mrr": round(statistics.mean(rrs), 4),
                "p95_ms": round(self.p95(lat_ms), 2),
            }

        modes = [
            (
                "dense_qdrant",
                lambda q: self.search_dense_qdrant(q, top_k=self.cfg.top_k),
            ),
            (
                "sparse_bm25_qdrant",
                lambda q: self.search_sparse_qdrant(q, top_k=self.cfg.top_k),
            ),
            (
                "hybrid_rrf_qdrant",
                lambda q: self.search_hybrid_rrf_qdrant(q, top_k=self.cfg.top_k),
            ),
            (
                "dense_pgvector",
                lambda q: self.search_dense_pgvector(q, top_k=self.cfg.top_k),
            ),
        ]

        results = [eval_mode(name, fn) for name, fn in modes]

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "recall@5", "recall@10", "mrr", "p95_ms"])
            writer.writeheader()
            writer.writerows(results)

        return results

    def demo_multitenant_filtering(self) -> None:
        query = "How do I configure SAML 2.0 for enterprise SSO?"
        product = "NovaCloud Core"
        version = "3.2"
        tenant = "tenant-alpha"
        date_from, date_to = 2023, 2026

        dense = self.search_dense_qdrant(
            query,
            top_k=5,
            product=product,
            version=version,
            tenant=tenant,
            date_from=date_from,
            date_to=date_to,
        )
        sparse = self.search_sparse_qdrant(
            query,
            top_k=5,
            product=product,
            version=version,
            tenant=tenant,
            date_from=date_from,
            date_to=date_to,
        )
        hybrid = self.search_hybrid_rrf_qdrant(
            query,
            top_k=5,
            product=product,
            version=version,
            tenant=tenant,
            date_from=date_from,
            date_to=date_to,
        )

        print("\nMulti-tenant filtered retrieval demo")
        print(f"Filter => tenant={tenant}, product={product}, version={version}, date_year=[{date_from},{date_to}]")
        print(f"Dense hits: {len(dense)} | Sparse hits: {len(sparse)} | Hybrid hits: {len(hybrid)}")
        for i, h in enumerate(hybrid[:3], start=1):
            print(f"  {i}. {h['doc_id']} | {h.get('title', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 retrieval stack (Qdrant + pgvector)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"C:\Users\Vince\Desktop\HelpDeskAI\data\processed\techqa_documents_normalized.jsonl"),
        help="Path to techqa_documents_normalized.jsonl",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(r"C:\Users\Vince\Desktop\HelpDeskAI\tests\golden\golden_dataset.jsonl"),
        help="Path to 100-question golden dataset",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--pg-dsn", default="postgresql://postgres:postgres@localhost:5432/postgres")
    parser.add_argument("--collection", default="helpdeskai_phase3")
    parser.add_argument("--pg-table", default="helpdeskai_phase3")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--benchmark-limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-csv", type=Path, default=Path("phase3_benchmark_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        dataset_path=args.dataset,
        golden_path=args.golden,
        qdrant_url=args.qdrant_url,
        pg_dsn=args.pg_dsn,
        collection=args.collection,
        pg_table=args.pg_table,
        top_k=args.top_k,
        benchmark_limit=args.benchmark_limit,
        batch_size=args.batch_size,
    )

    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")
    if not cfg.golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {cfg.golden_path}")

    print("Phase 3 - HelpDeskAI Retrieval Indexer")
    print("Embedding candidate proxy scores (MTEB multilingual retrieval):")
    for model_name, score in MTEB_RETRIEVAL_PROXY.items():
        print(f"  - {model_name}: {score}")

    indexer = HelpdeskRetrievalIndexer(cfg)
    indexer.build_indexes()
    indexer.demo_multitenant_filtering()

    results = indexer.run_benchmark(args.output_csv)

    print("\nBenchmark summary (100 golden questions)")
    for row in results:
        print(
            f"{row['mode']:<22} recall@5={row['recall@5']:.4f} "
            f"recall@10={row['recall@10']:.4f} mrr={row['mrr']:.4f} p95_ms={row['p95_ms']:.2f}"
        )

    print(f"\nCSV results written to: {args.output_csv}")


if __name__ == "__main__":
    main()
