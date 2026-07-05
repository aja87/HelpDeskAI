from __future__ import annotations

import logging

from argparse import ArgumentParser
from pathlib import Path

from helpdeskai.corpus.config import DATA_DIR, LOG_DIR, LOG_FILE, DownloadConfig
from helpdeskai.corpus.downloader import run_download
from helpdeskai.common.logging import init_logging


def parse_args() -> DownloadConfig:
    """Parse CLI options controlling output paths and subset sizes."""

    parser = ArgumentParser(description="Download and subset HelpDeskAI corpora")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--techqa-docs", type=int, default=5000)
    parser.add_argument("--techqa-qa", type=int, default=600)
    parser.add_argument("--bitext", type=int, default=2000)
    parser.add_argument("--msdialog", type=int, default=500)
    return DownloadConfig(**vars(parser.parse_args()))


def main() -> None:
    """CLI runner for corpus download."""

    config = parse_args()
    init_logging(log_dir=LOG_DIR, log_file=LOG_FILE, level=logging.INFO)
    run_download(config)

if __name__ == "__main__":
    main()
