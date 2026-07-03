from __future__ import annotations

import argparse
from pathlib import Path

from helpdeskai.ingestion.config import GOLDEN_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, IngestionConfig
from helpdeskai.ingestion.workflow import run_ingestion_core


def parse_args() -> IngestionConfig:
    """Parse CLI arguments and return the ingestion configuration."""

    parser = argparse.ArgumentParser(description="Run the HelpDeskAI phase-2 ingestion pipeline")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-sample-size", type=int, default=50)
    parser.add_argument("--golden-size", type=int, default=100)
    return IngestionConfig(**vars(parser.parse_args()))


def main() -> None:
    """CLI entrypoint for local ingestion runs."""

    run_ingestion_core(parse_args())


if __name__ == "__main__":
    main()
