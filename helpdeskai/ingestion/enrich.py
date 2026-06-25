"""Metadata enrichment stage for canonical TechQA documents."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from dateutil import parser as date_parser

METADATA_FIELDS = ("category", "product", "versions", "date")

CATEGORY_PATTERNS = (
    ("technote_troubleshooting", re.compile(r"\bTECHNOTE\s+\(TROUBLESHOOTING\)\s*$", re.I)),
    ("technote_faq", re.compile(r"\bTECHNOTE\s+\(FAQ\)\s*$", re.I)),
    ("security_bulletin", re.compile(r"\bSECURITY\s+BULLETIN\s*$", re.I)),
    ("flash_alert", re.compile(r"\bFLASH\s+\(ALERT\)\s*$", re.I)),
    ("product_documentation", re.compile(r"\bPRODUCT\s+DOCUMENTATION\s*$", re.I)),
    ("downloadable_files", re.compile(r"\bDOWNLOADABLE\s+FILES?\s*$", re.I)),
    ("release_notes", re.compile(r"\bRELEASE\s+NOTES\s*$", re.I)),
    ("white_paper", re.compile(r"\bWHITE\s+PAPER\s*$", re.I)),
    ("fix_readme", re.compile(r"\bFIX\s+README\s*$", re.I)),
    ("fix_available", re.compile(r"\b(?:A\s+)?FIX(?:ES)?\s+(?:ARE|IS)\s+AVAILABLE\s*$", re.I)),
)
KNOWN_PRODUCT_PATTERNS = (
    re.compile(r"\bIBM Streams\b", re.I),
    re.compile(r"\bIBM HTTP Server\b", re.I),
    re.compile(r"\bIBM Integration Bus\b", re.I),
    re.compile(r"\bIBM MQ Appliance\b", re.I),
    re.compile(r"\bIBM MQ\b", re.I),
    re.compile(r"\bWebSphere Application Server\b", re.I),
    re.compile(r"\bWebSphere Message Broker\b", re.I),
    re.compile(r"\bWebSphere MQ\b", re.I),
    re.compile(r"\bWebSphere Portal\b", re.I),
    re.compile(r"\bContent Platform Engine\b", re.I),
    re.compile(r"\bIBM SPSS Statistics\b", re.I),
    re.compile(r"\bSPSS Statistics\b", re.I),
    re.compile(r"\bRational Application Developer\b", re.I),
    re.compile(r"\bTivoli Integrated Portal\b", re.I),
    re.compile(r"\bTivoli Network Manager IP Edition\b", re.I),
    re.compile(r"\bTivoli Monitoring\b", re.I),
    re.compile(r"\bDataPower(?: SOA Appliance)?\b", re.I),
)
EXPLICIT_PRODUCT_RE = re.compile(
    r"(?im)^\s*(?:product|software)\s*[:=-]\s*(?P<value>[^\n]{2,120})\s*$"
)
EXPLICIT_VERSION_RE = re.compile(
    r"(?im)^\s*(?:product\s+)?version(?:s)?\s*[:=-]\s*(?P<value>[^\n]{1,120})\s*$"
)
TITLE_VERSION_RE = re.compile(
    r"(?ix)(?:\bversion\b|\brelease\b|\bfirmware\b|\bfix\s+pack\b|\bfp\b|\bv)"
    r"\s*[:=-]?\s*(?P<version>\d+(?:\.\d+){1,4}(?:[-_A-Za-z0-9]+)?)"
)
SEMVER_RE = re.compile(r"(?<![\w.-])v?(?P<version>\d+(?:\.\d+){1,4})(?![\w.-])", re.I)
DATE_LINE_RE = re.compile(
    r"(?im)^\s*(?P<label>publication date|published|last updated|updated|modified)"
    r"\s*[:=-]\s*(?P<value>[^\n]{4,60})\s*$"
)
SAFE_DATE_RE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4})$",
    re.I,
)


def _metadata_value(
    value: Any = None,
    *,
    method: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    return {"value": value, "method": method, "confidence": confidence}


def _extract_category(header: str) -> tuple[dict[str, Any], str]:
    for category, pattern in CATEGORY_PATTERNS:
        match = pattern.search(header)
        if match:
            return (
                _metadata_value(category, method="header_suffix", confidence=1.0),
                header[: match.start()].rstrip(" ;-"),
            )
    return _metadata_value(), header


def _extract_title_country_keywords(header: str) -> tuple[str, str | None, list[str]]:
    marker = " - United States"
    marker_index = header.find(marker)
    if marker_index == -1:
        title, tail, country = header, "", None
    else:
        title = header[:marker_index]
        tail = header[marker_index + len(marker) :]
        country = "United States"
    title = re.sub(r"^\s*IBM\s+", "", title, flags=re.I).strip(" ;-")
    keywords = []
    for keyword in tail.strip(" ;-").split(";"):
        cleaned = keyword.strip()
        if cleaned and cleaned.casefold() not in {item.casefold() for item in keywords}:
            keywords.append(cleaned)
    return title, country, keywords


def _extract_product(text: str, title: str) -> dict[str, Any]:
    explicit = EXPLICIT_PRODUCT_RE.search(text)
    if explicit:
        return _metadata_value(
            explicit.group("value").strip(" .;-"),
            method="explicit_field",
            confidence=1.0,
        )
    for pattern in KNOWN_PRODUCT_PATTERNS:
        match = pattern.search(title)
        if match:
            return _metadata_value(
                match.group(0),
                method="known_product_in_title",
                confidence=0.9,
            )
    return _metadata_value()


def _version_values(value: str) -> list[str]:
    return list(dict.fromkeys(match.group("version") for match in SEMVER_RE.finditer(value)))


def _extract_versions(text: str, title: str) -> dict[str, Any]:
    explicit = EXPLICIT_VERSION_RE.search(text)
    if explicit and (versions := _version_values(explicit.group("value"))):
        return _metadata_value(versions, method="explicit_field", confidence=1.0)
    versions = list(
        dict.fromkeys(match.group("version") for match in TITLE_VERSION_RE.finditer(title))
    )
    if versions:
        return _metadata_value(versions, method="marked_version_in_title", confidence=0.9)
    return _metadata_value()


def _extract_date(text: str) -> dict[str, Any]:
    for match in DATE_LINE_RE.finditer(text):
        raw_value = match.group("value").strip(" .")
        if not SAFE_DATE_RE.fullmatch(raw_value):
            continue
        try:
            parsed = date_parser.parse(raw_value, fuzzy=False).date()
        except (OverflowError, ValueError):
            continue
        label = match.group("label").lower().replace(" ", "_")
        return _metadata_value(
            parsed.isoformat(),
            method=f"explicit_{label}",
            confidence=1.0,
        )
    return _metadata_value()


def extract_metadata(text: str) -> dict[str, Any]:
    """Extract conservative metadata from normalized TechQA text."""
    header = next((line.strip() for line in text.splitlines() if line.strip()), "")
    category, header_without_category = _extract_category(header)
    title, country, keywords = _extract_title_country_keywords(header_without_category)
    return {
        "title": title or None,
        "country": country,
        "keywords": keywords,
        "category": category,
        "product": _extract_product(text, title),
        "versions": _extract_versions(text, title),
        "date": _extract_date(text),
    }


def enrich_documents(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach conservative, traceable metadata to canonical documents."""
    enriched = []
    for document in documents:
        item = dict(document)
        item["metadata"] = extract_metadata(item["text"])
        enriched.append(item)
    return enriched


def metadata_coverage(documents: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count populated index-filter metadata fields."""
    return {
        field: sum(
            document["metadata"][field]["value"] is not None for document in documents
        )
        for field in METADATA_FIELDS
    }
