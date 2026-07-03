"""Index processed TechQA chunks into Qdrant and pgvector."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.retrieval.indexer import index_corpus  # noqa: E402
from helpdeskai.retrieval.models import RetrievalConfig  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/processed/techqa/chunks.jsonl"),
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--pgvector-dsn",
        default="postgresql://postgres:postgres@localhost:5433/helpdeskai",
    )
    parser.add_argument("--collection", default="helpdeskai_techqa_chunks")
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-qdrant", action="store_true")
    parser.add_argument("--no-pgvector", action="store_true")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RetrievalConfig(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        pgvector_dsn=args.pgvector_dsn,
        model_name=args.model_name,
        corpus_path=args.corpus_path,
        batch_size=args.batch_size,
    )
    count = index_corpus(
        config=config,
        recreate=not args.append,
        index_qdrant=not args.no_qdrant,
        index_pgvector=not args.no_pgvector,
        progress=print,
    )
    print(f"Done. Indexed {count} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
