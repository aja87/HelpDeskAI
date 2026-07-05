from __future__ import annotations

import argparse
import os
import logging

from pathlib import Path
from helpdeskai.common.logging import init_logging
from helpdeskai.indexing.config import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_MANIFEST_PATH,
    LOG_FILE,
    IndexingConfig,
)
from helpdeskai.indexing.workflow import run_indexing_core


def parse_args() -> IndexingConfig:
    """Parse CLI arguments and return indexation configuration."""

    parser = argparse.ArgumentParser(description="Run the HelpDeskAI phase-3 Qdrant indexation")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection-name", type=str, default="helpdeskai-techqa")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=128)
    parser.add_argument(
        "--recreate-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop and recreate collection before indexing",
    )
    return IndexingConfig(**vars(parser.parse_args()))


def main() -> None:
    """CLI entrypoint for local Qdrant indexation runs."""

    init_logging(log_file=LOG_FILE)
    manifest = run_indexing_core(parse_args())
    counts = manifest["counts"]
    logging.info(
        "Indexed chunks into Qdrant: "
        f"loaded={counts['chunks_loaded']} embedded={counts['chunks_embedded']} indexed={counts['chunks_indexed']}"
    )

if __name__ == "__main__":
    main()