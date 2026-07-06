"""TechQA extraction stage with a normalized payload contract."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any

from bs4 import BeautifulSoup

from helpdeskai.ingestion.exceptions import TechQAIngestionError

HTML_TAG_RE = re.compile(
    r"</?(?:html|body|head|title|meta|script|style|div|p|br|table|tr|td|th|"
    r"ul|ol|li|pre|code|h[1-6]|a|span|strong|em)\b[^>]*>",
    re.IGNORECASE,
)


def _repair_common_mojibake(text: str) -> str:
    markers = ("Ãƒ", "Ã‚", "Ã¢â‚¬", "Ã¢â‚¬â„¢", "Ã¢â‚¬Å“", "Ã¢â‚¬â€œ", "ï¿½")
    before_score = sum(text.count(marker) for marker in markers)
    if before_score == 0:
        return text
    try:
        candidate = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    after_score = sum(candidate.count(marker) for marker in markers)
    return candidate if after_score < before_score else text


def extract_text(raw_document: str) -> tuple[str, bool]:
    """Convert HTML-like content to text while preserving useful line breaks."""
    if not isinstance(raw_document, str):
        raise TechQAIngestionError("TechQA document content must be a string")
    repaired = _repair_common_mojibake(raw_document)
    contained_html = bool(HTML_TAG_RE.search(repaired))
    if contained_html:
        soup = BeautifulSoup(repaired, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        repaired = soup.get_text("\n")
    else:
        repaired = html.unescape(repaired)
    return repaired, contained_html


def extract_techqa_documents(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract raw TechQA records into stage payloads."""
    payloads = []
    for index, record in enumerate(records):
        missing = {"id", "split", "document"}.difference(record)
        if missing:
            raise TechQAIngestionError(
                f"Document {index} is missing fields: {', '.join(sorted(missing))}"
            )
        text, contained_html = extract_text(record["document"])
        payloads.append(
            {
                "source_id": str(record["id"]),
                "split": str(record["split"]),
                "text": text,
                "extraction_method": "beautifulsoup" if contained_html else "plain_text",
                "source_type": "html" if contained_html else "text",
                "contained_html": contained_html,
                "status": "ok" if text.strip() else "empty",
            }
        )
    return payloads
