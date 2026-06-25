"""Stage-oriented Prefect pipeline for TechQA corpus preparation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from prefect import flow, task

from helpdeskai.ingestion.chunk import HuggingFaceTokenizer, chunk_documents
from helpdeskai.ingestion.dedup import deduplicate_chunks, deduplicate_documents
from helpdeskai.ingestion.enrich import enrich_documents
from helpdeskai.ingestion.exceptions import TechQAIngestionError
from helpdeskai.ingestion.extract import extract_techqa_documents
from helpdeskai.ingestion.io import read_jsonl
from helpdeskai.ingestion.normalize import normalize_documents
from helpdeskai.ingestion.persist import (
    build_ingestion_manifest,
    persist_ingestion,
)
from helpdeskai.ingestion.quality import (
    generate_evidently_report,
    validate_chunks,
)


@task
def load_raw_techqa_task(raw_dir: Path) -> list[dict]:
    """Load the downloaded TechQA document corpus."""
    return read_jsonl(raw_dir / "techqa/documents.jsonl")


@task
def extract_documents_task(records: list[dict]) -> list[dict[str, Any]]:
    """Extract text and extraction provenance from raw TechQA records."""
    return extract_techqa_documents(records)


@task
def normalize_documents_task(payloads: list[dict]) -> list[dict[str, Any]]:
    """Normalize extracted document payloads."""
    return normalize_documents(payloads)


@task
def deduplicate_documents_task(
    payloads: list[dict],
) -> list[dict[str, Any]]:
    """Create canonical documents."""
    return deduplicate_documents(payloads)


@task
def enrich_documents_task(documents: list[dict]) -> list[dict[str, Any]]:
    """Attach traceable product, version, date, and category metadata."""
    return enrich_documents(documents)


@task
def chunk_documents_task(documents: list[dict]) -> list[dict[str, Any]]:
    """Apply the selected recursive chunking strategy."""
    return chunk_documents(documents, HuggingFaceTokenizer())


@task
def deduplicate_chunks_task(chunks: list[dict]) -> list[dict[str, Any]]:
    """Remove exact duplicate chunks before persistence."""
    return deduplicate_chunks(chunks)


@task
def persist_corpus_task(
    raw_dir: Path,
    processed_dir: Path,
    raw_documents: list[dict],
    extracted: list[dict],
    documents: list[dict],
    chunks: list[dict],
    force: bool,
) -> Path:
    """Atomically publish normalized ingestion outputs."""
    manifest = build_ingestion_manifest(
        documents,
        chunks,
        source_document_count=len(raw_documents),
        html_source_count=sum(item["contained_html"] for item in extracted),
        documents_path=raw_dir / "techqa/documents.jsonl",
    )
    return persist_ingestion(
        processed_dir,
        documents,
        chunks,
        manifest,
        force=force,
    )


@task
def quality_report_task(
    chunks: list[dict],
    document_ids: set[str],
    report_path: Path,
    force: bool,
) -> Path:
    """Validate final chunks and atomically publish Evidently outputs."""
    if report_path.exists() and not force:
        raise TechQAIngestionError(
            "Corpus quality report exists. Use --force to replace it"
        )
    quality = validate_chunks(chunks, document_ids)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".corpus-quality-",
        dir=report_path.parent,
    ) as temporary:
        staging = Path(temporary)
        staged_report = staging / report_path.name
        staged_summary = staging / "corpus_quality_summary.json"
        generate_evidently_report(chunks, staged_report)
        staged_summary.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path = report_path.parent / "corpus_quality_summary.json"
        for source, destination in (
            (staged_report, report_path),
            (staged_summary, summary_path),
        ):
            if destination.exists():
                destination.unlink()
            source.replace(destination)
    return report_path


@flow(name="helpdeskai-corpus-preparation", log_prints=True)
def corpus_preparation_flow(
    *,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed/techqa"),
    report_path: Path = Path("docs/corpus_preparation/corpus_quality_report.html"),
    force: bool = False,
    skip_existing: bool = False,
) -> Path:
    """Run extract, normalize, deduplicate, enrich, persist, and quality stages."""
    expected_outputs = (
        processed_dir / "documents.jsonl",
        processed_dir / "chunks.jsonl",
        processed_dir / "manifest.json",
        report_path,
        report_path.parent / "corpus_quality_summary.json",
    )
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if skip_existing:
        if len(existing_outputs) == len(expected_outputs):
            return report_path
        if existing_outputs:
            raise TechQAIngestionError(
                "Corpus-preparation outputs are incomplete; use --force to regenerate them"
            )

    raw_documents = load_raw_techqa_task(raw_dir)
    extracted = extract_documents_task(raw_documents)
    normalized = normalize_documents_task(extracted)
    canonical = deduplicate_documents_task(normalized)
    enriched = enrich_documents_task(canonical)
    chunks = chunk_documents_task(enriched)
    unique_chunks = deduplicate_chunks_task(chunks)
    persist_corpus_task(
        raw_dir,
        processed_dir,
        raw_documents,
        extracted,
        enriched,
        unique_chunks,
        force,
    )
    return quality_report_task(
        unique_chunks,
        {document["document_id"] for document in enriched},
        report_path,
        force,
    )


__all__ = [
    "corpus_preparation_flow",
    "chunk_documents_task",
    "deduplicate_chunks_task",
    "deduplicate_documents_task",
    "enrich_documents_task",
    "extract_documents_task",
    "load_raw_techqa_task",
    "normalize_documents_task",
    "persist_corpus_task",
    "quality_report_task",
]
