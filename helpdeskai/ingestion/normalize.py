"""Text normalization stage for extracted TechQA payloads."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from helpdeskai.ingestion.exceptions import TechQAIngestionError

NORMALIZATION_VERSION = "1.0"
WHITESPACE_RE = re.compile(r"[^\S\n]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
)


def normalize_text(text: str) -> str:
    """Normalize Unicode, punctuation, spacing, and line endings."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(PUNCTUATION_TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.splitlines():
        cleaned = WHITESPACE_RE.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
        elif lines and lines[-1] != "":
            lines.append("")
    normalized = "\n".join(lines).strip()
    return BLANK_LINES_RE.sub("\n\n", normalized)


def clean_document(raw_document: str) -> tuple[str, bool]:
    """Extract and normalize a raw TechQA document."""
    from helpdeskai.ingestion.extract import extract_text

    extracted, contained_html = extract_text(raw_document)
    cleaned = normalize_text(extracted)
    if not cleaned:
        raise TechQAIngestionError("TechQA document is empty after normalization")
    return cleaned, contained_html


def normalize_documents(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize extracted text while preserving extraction provenance."""
    normalized = []
    for payload in payloads:
        item = dict(payload)
        item["text"] = normalize_text(str(item.get("text", "")))
        if not item["text"]:
            raise TechQAIngestionError(
                f"Document {item.get('source_id')} is empty after normalization"
            )
        item["normalized"] = True
        item["normalization_version"] = NORMALIZATION_VERSION
        normalized.append(item)
    return normalized
