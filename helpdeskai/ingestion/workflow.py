"""Ingestion workflow orchestration and optional Prefect integration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

from .chunking import benchmark_chunking_strategies, chunk_documents
from .config import IngestionConfig
from .golden import build_golden_dataset
from .io_utils import read_jsonl, write_json, write_jsonl
from .normalize import prepare_msdialog_conversations, prepare_qa_pairs, prepare_techqa_documents
from .quality import build_quality_report


try:
    from prefect import flow, task
except ImportError:  # pragma: no cover - exercised only when Prefect is absent.
    F = TypeVar("F", bound=Callable[..., Any])

    def _passthrough_decorator(fn: F | None = None, **_: Any) -> Callable[[F], F] | F:
        if fn is not None:
            return fn

        def wrapper(inner: F) -> F:
            return inner

        return wrapper

    flow = _passthrough_decorator
    task = _passthrough_decorator


@task
def load_raw_corpora(config: IngestionConfig) -> dict[str, list[dict[str, Any]]]:
    """Load all raw phase-2 corpora from disk."""

    return {
        "techqa_documents": read_jsonl(config.raw_dir / "techqa_documents.jsonl"),
        "techqa_qa": read_jsonl(config.raw_dir / "techqa_qa.jsonl"),
        "bitext_pairs": read_jsonl(config.raw_dir / "bitext_pairs.jsonl"),
        "msdialog": read_jsonl(config.raw_dir / "msdialog_conversations.jsonl"),
    }


@task
def normalize_corpora(raw: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize every corpus into deterministic downstream schemas."""

    return {
        "documents": prepare_techqa_documents(raw["techqa_documents"]),
        "techqa_qa": prepare_qa_pairs(raw["techqa_qa"], source="techqa"),
        "bitext_pairs": prepare_qa_pairs(raw["bitext_pairs"], source="bitext"),
        "msdialog": prepare_msdialog_conversations(raw["msdialog"]),
    }


@task
def persist_outputs(
    normalized: dict[str, list[dict[str, Any]]],
    chunk_rows: list[dict[str, Any]],
    chunk_report: dict[str, Any],
    quality_summary: dict[str, Any],
    config: IngestionConfig,
) -> dict[str, str]:
    """Persist normalized corpora, chunks, summaries, and manifest files."""

    processed_dir = config.processed_dir
    reports_dir = config.reports_dir
    golden_dir = config.golden_dir

    normalized_documents_path = processed_dir / "techqa_documents_normalized.jsonl"
    normalized_techqa_qa_path = processed_dir / "techqa_qa_normalized.jsonl"
    normalized_bitext_path = processed_dir / "bitext_pairs_normalized.jsonl"
    normalized_msdialog_path = processed_dir / "msdialog_conversations_normalized.jsonl"
    chunk_path = processed_dir / "techqa_chunks.jsonl"
    chunk_report_path = reports_dir / "chunking_benchmark.json"
    quality_summary_path = reports_dir / "quality_summary.json"
    manifest_path = processed_dir / "ingestion_manifest.json"
    golden_path = golden_dir / "golden_dataset.jsonl"

    write_jsonl(normalized_documents_path, normalized["documents"])
    write_jsonl(normalized_techqa_qa_path, normalized["techqa_qa"])
    write_jsonl(normalized_bitext_path, normalized["bitext_pairs"])
    write_jsonl(normalized_msdialog_path, normalized["msdialog"])
    write_jsonl(chunk_path, chunk_rows)
    write_json(chunk_report_path, chunk_report)
    write_json(quality_summary_path, quality_summary)

    build_golden_dataset(
        normalized["techqa_qa"],
        normalized["bitext_pairs"],
        golden_path,
        target_size=config.golden_size,
        seed=config.seed,
    )

    manifest = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "artifacts": {
            "documents": str(normalized_documents_path),
            "techqa_qa": str(normalized_techqa_qa_path),
            "bitext_pairs": str(normalized_bitext_path),
            "msdialog": str(normalized_msdialog_path),
            "chunks": str(chunk_path),
            "chunk_benchmark": str(chunk_report_path),
            "quality_summary": str(quality_summary_path),
            "golden_dataset": str(golden_path),
        },
        "counts": {
            "documents": len(normalized["documents"]),
            "techqa_qa": len(normalized["techqa_qa"]),
            "bitext_pairs": len(normalized["bitext_pairs"]),
            "msdialog": len(normalized["msdialog"]),
            "chunks": len(chunk_rows),
        },
    }
    write_json(manifest_path, manifest)

    return {"manifest": str(manifest_path), "golden_dataset": str(golden_path)}


def run_ingestion_core(config: IngestionConfig) -> dict[str, str]:
    """Execute ingestion without requiring a Prefect runtime."""

    raw = load_raw_corpora.fn(config) if hasattr(load_raw_corpora, "fn") else load_raw_corpora(config)
    normalized = (
        normalize_corpora.fn(raw) if hasattr(normalize_corpora, "fn") else normalize_corpora(raw)
    )
    chunk_report = benchmark_chunking_strategies(normalized["documents"], config)
    strategy_name = chunk_report["recommended_strategy"]
    chunk_rows = chunk_documents(normalized["documents"], strategy_name, config)
    quality_summary = build_quality_report(
        normalized["documents"], config.reports_dir / "data_quality.html"
    )
    if hasattr(persist_outputs, "fn"):
        return persist_outputs.fn(normalized, chunk_rows, chunk_report, quality_summary, config)
    return persist_outputs(normalized, chunk_rows, chunk_report, quality_summary, config)


@flow(name="helpdeskai-ingestion")
def run_ingestion_flow(config: IngestionConfig | None = None) -> dict[str, str]:
    """Run the full phase-2 ingestion pipeline."""

    config = config or IngestionConfig()
    return run_ingestion_core(config)
