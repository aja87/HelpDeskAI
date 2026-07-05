from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from helpdeskai.common.logging import init_logging
from helpdeskai.retrieval.config import (
    DEFAULT_CHUNKS_PATH,
    LOG_FILE,
    RetrievalConfig,
    VALID_RETRIEVAL_MODES,
)
from helpdeskai.retrieval.workflow import SearchFilters, run_retrieval_core


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retrieval search."""

    parser = argparse.ArgumentParser(description="Run retrieval search for local developer checks")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection-name", type=str, default="helpdeskai-techqa")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--mode", type=str, default="hybrid", choices=sorted(VALID_RETRIEVAL_MODES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query",type=str,default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--product", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        chunks_path=args.chunks_path,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        mode=args.mode,
        top_k=args.top_k,
    )


def _build_filters(args: argparse.Namespace) -> SearchFilters | None:
    has_filters = any(
        [
            args.source,
            args.product,
            args.version,
            args.category,
            args.date_from,
            args.date_to,
        ]
    )
    if not has_filters:
        return None

    return SearchFilters(
        source=args.source,
        product=args.product,
        version=args.version,
        category=args.category,
        date_from=args.date_from,
        date_to=args.date_to,
    )

def main() -> None:
    args = parse_args()
    init_logging(log_file=LOG_FILE)
    payload = run_retrieval_core(
        _build_config(args),
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        filters=_build_filters(args),
    )
    logging.info("Query: %s", payload["query"])
    logging.info("Mode: %s | Top-k: %s", payload["mode"], payload["top_k"])
    for index, row in enumerate(payload["results"], start=1):
        logging.info(
            "[%d] chunk_id=%s score=%.4f source=%s product=%s",
            index,
            row["chunk_id"],
            row["score"],
            row["source"],
            row["product"],
        )
        logging.info("    %s", row["text"][:220].replace("\n", " "))


if __name__ == "__main__":
    main()