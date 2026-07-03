"""Compatibility facade for ingestion utilities.

This module preserves previous import paths while delegating implementation to
smaller modules under ``helpdeskai.ingestion``.
"""

from .chunking import benchmark_chunking_strategies, chunk_documents
from .config import IngestionConfig
from .golden import build_golden_dataset
from .normalize import normalize_text, prepare_msdialog_conversations, prepare_qa_pairs, prepare_techqa_documents
from .workflow import run_ingestion_core, run_ingestion_flow

__all__ = [
    "IngestionConfig",
    "benchmark_chunking_strategies",
    "build_golden_dataset",
    "chunk_documents",
    "normalize_text",
    "prepare_msdialog_conversations",
    "prepare_qa_pairs",
    "prepare_techqa_documents",
    "run_ingestion_core",
    "run_ingestion_flow",
]
