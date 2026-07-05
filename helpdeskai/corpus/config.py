"""Configuration and constants for corpus download workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_DIR = Path("data/raw")
LOG_FILE = "corpus.log"
CHECKSUM_FILE = "checksums.sha256.json"

TECHQA_REPO = "rojagtap/tech-qa"
BITEXT_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MSDIALOG_URL = (
    "https://raw.githubusercontent.com/SCU-ChenYue/MSDialog_RL/main/test_MSDialog.jsonl"
)


@dataclass(slots=True)
class DownloadConfig:
    """Runtime settings used to fetch and build raw corpus subsets."""

    output_dir: Path = DATA_DIR
    seed: int = 42
    overwrite: bool = False
    techqa_docs: int = 5000
    techqa_qa: int = 600
    bitext: int = 2000
    msdialog: int = 500
