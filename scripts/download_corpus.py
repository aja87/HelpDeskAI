from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from helpdeskai.corpus.config import DATA_DIR, LOG_DIR, LOG_FILE, DownloadConfig
from helpdeskai.corpus.downloader import run_download


def parse_args() -> DownloadConfig:
    """Parse CLI options controlling output paths and subset sizes."""

    parser = argparse.ArgumentParser(description="Download and subset HelpDeskAI corpora")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--techqa-docs", type=int, default=5000)
    parser.add_argument("--techqa-qa", type=int, default=600)
    parser.add_argument("--bitext", type=int, default=2000)
    parser.add_argument("--msdialog", type=int, default=500)
    return DownloadConfig(**vars(parser.parse_args()))


def configure_logging(log_dir: Path = LOG_DIR) -> None:
    """Configure file and stdout logging used by corpus download scripts."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, LOG_FILE)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """CLI runner for corpus download."""

    config = parse_args()
    configure_logging()
    run_download(config)

if __name__ == "__main__":
    main()
