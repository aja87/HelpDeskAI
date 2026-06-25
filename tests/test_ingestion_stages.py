from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpdeskai.ingestion.dedup import deduplicate_documents
from helpdeskai.ingestion.enrich import enrich_documents, metadata_coverage
from helpdeskai.ingestion.exceptions import TechQAIngestionError
from helpdeskai.ingestion.extract import extract_techqa_documents
from helpdeskai.ingestion.normalize import normalize_documents
from helpdeskai.ingestion.persist import build_ingestion_manifest, persist_ingestion


def raw_documents() -> list[dict]:
    return [
        {
            "id": "Q1",
            "split": "train",
            "document": (
                "<h1>IBM WebSphere Application Server - United States TECHNOTE (FAQ)</h1>"
                "<p>Published: January 23, 2018</p><p>Useful answer.</p>"
            ),
        },
        {
            "id": "Q2",
            "split": "validation",
            "document": (
                "IBM WebSphere Application Server - United States TECHNOTE (FAQ)\n"
                "Published: January 23, 2018\nUseful answer."
            ),
        },
    ]


def test_extract_normalize_dedup_enrich_contract() -> None:
    extracted = extract_techqa_documents(raw_documents())
    normalized = normalize_documents(extracted)
    canonical = deduplicate_documents(normalized)
    enriched = enrich_documents(canonical)

    assert extracted[0]["extraction_method"] == "beautifulsoup"
    assert normalized[0]["normalized"] is True
    assert len(canonical) == 1
    assert enriched[0]["source_ids"] == ["Q1", "Q2"]
    assert enriched[0]["metadata"]["product"]["value"] == "WebSphere Application Server"
    assert metadata_coverage(enriched)["category"] == 1


def test_manifest_and_atomic_persistence(tmp_path: Path) -> None:
    extracted = extract_techqa_documents(raw_documents())
    normalized = normalize_documents(extracted)
    canonical = deduplicate_documents(normalized)
    enriched = enrich_documents(canonical)
    manifest = build_ingestion_manifest(
        enriched,
        [],
        source_document_count=2,
        html_source_count=1,
        documents_path=Path("data/raw/techqa/documents.jsonl"),
    )
    output = tmp_path / "processed"

    manifest_path = persist_ingestion(
        output,
        enriched,
        [],
        manifest,
        force=False,
    )

    assert manifest_path.is_file()
    assert len((output / "documents.jsonl").read_text().splitlines()) == 1
    assert (output / "chunks.jsonl").is_file()
    assert json.loads(manifest_path.read_text())["removed_duplicate_rows"] == 1
    with pytest.raises(TechQAIngestionError, match="--force"):
        persist_ingestion(output, enriched, [], manifest, force=False)
