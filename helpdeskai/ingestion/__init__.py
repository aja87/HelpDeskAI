"""TechQA extraction, normalization, enrichment, persistence, and quality utilities."""

from helpdeskai.ingestion.chunk import chunk_documents
from helpdeskai.ingestion.dedup import deduplicate_chunks, deduplicate_documents
from helpdeskai.ingestion.enrich import enrich_documents, extract_metadata
from helpdeskai.ingestion.exceptions import TechQAIngestionError
from helpdeskai.ingestion.extract import extract_techqa_documents
from helpdeskai.ingestion.normalize import clean_document, normalize_documents, normalize_text
from helpdeskai.ingestion.persist import persist_ingestion

__all__ = [
    "TechQAIngestionError",
    "clean_document",
    "chunk_documents",
    "deduplicate_chunks",
    "deduplicate_documents",
    "enrich_documents",
    "extract_techqa_documents",
    "extract_metadata",
    "normalize_text",
    "normalize_documents",
    "persist_ingestion",
]
