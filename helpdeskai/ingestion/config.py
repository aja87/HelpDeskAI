"""Configuration definitions for the ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/ingestion")
GOLDEN_DIR = Path("tests/golden")
LOG_FILE = "ingestion.log"


@dataclass(slots=True)
class IngestionConfig:
    """Runtime configuration for the phase-2 ingestion pipeline."""

    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    reports_dir: Path = REPORTS_DIR
    golden_dir: Path = GOLDEN_DIR
    seed: int = 42
    chunk_sample_size: int = 50
    golden_size: int = 100
    fixed_chunk_size: int = 1200
    fixed_overlap: int = 120
    recursive_chunk_size: int = 1000
    recursive_overlap: int = 120
    semantic_chunk_size: int = 900
