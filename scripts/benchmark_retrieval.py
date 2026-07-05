from __future__ import annotations

from argparse import Namespace, ArgumentParser
import logging
import os
from pathlib import Path

from helpdeskai.common.logging import init_logging
from helpdeskai.retrieval.config import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_CHUNKS_PATH,
    GOLDEN_PATH,
    LOG_FILE,
    RetrievalConfig,
)
from helpdeskai.retrieval.workflow import run_benchmark_core


def parse_args() -> Namespace:
    """Parse CLI args for retrieval benchmark."""

    parser = ArgumentParser(description="Run retrieval benchmark and save report")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--golden-path", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection-name", type=str, default="helpdeskai-techqa")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def build_config(args: Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        chunks_path=args.chunks_path,
        golden_path=args.golden_path,
        benchmark_path=args.output_path,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        benchmark_sample_size=args.sample_size,
    )


def main() -> None:
    args = parse_args()
    init_logging(log_file=LOG_FILE)
    report = run_benchmark_core(
        build_config(args),
        sample_size=args.sample_size,
        top_k=args.top_k,
        output_path=args.output_path,
    )

    logging.info("Benchmark written to %s", args.output_path)
    for mode, metrics in report["metrics"].items():
        logging.info(
            "%s | recall@5=%.4f recall@10=%.4f mrr=%.4f p95=%.2fms",
            mode,
            metrics["recall_at_5"],
            metrics["recall_at_10"],
            metrics["mrr"],
            metrics["latency_p95_ms"],
        )


if __name__ == "__main__":
    main()
