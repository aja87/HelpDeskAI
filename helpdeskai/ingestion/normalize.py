"""Text normalization and schema preparation for raw corpora."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
VERSION_RE = re.compile(r"\b\d+(?:\.\d+){1,3}\b")

CATEGORY_MARKERS = {
    "TROUBLESHOOTING": "troubleshooting",
    "SECURITY BULLETIN": "security_bulletin",
    "TECHNOTE": "technote",
    "DOWNLOAD": "download",
    "FAQ": "faq",
    "INSTALLATION": "installation",
    "CONFIGURATION": "configuration",
}


def normalize_text(text: str) -> str:
    """Clean HTML fragments and normalize spacing for downstream indexing."""

    if not text:
        return ""
    cleaned = html.unescape(text)
    cleaned = cleaned.replace("[SEP]", "\n")
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = BLANK_LINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def normalize_label(value: str) -> str:
    """Normalize label-like metadata into lowercase snake case."""

    cleaned = normalize_text(value)
    if not cleaned:
        return ""
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def stable_checksum(text: str) -> str:
    """Build a deterministic checksum for deduplication and provenance."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_title(text: str) -> str:
    """Use the first non-empty line as a fallback title."""

    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:160]
    return ""


def infer_product(text: str) -> str:
    """Extract a rough product name from the first line of TechQA documents."""

    first_line = infer_title(text)
    patterns = [
        r"^IBM\s+(.*?)\s+-\s+United States",
        r"^IBM\s+(.*?)\s+(?:TECHNOTE|SECURITY BULLETIN|FAQ|DOWNLOAD)",
    ]
    for pattern in patterns:
        match = re.search(pattern, first_line, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1))
    return ""


def infer_version(text: str) -> str:
    """Extract a likely version string from a document body."""

    match = VERSION_RE.search(text)
    return match.group(0) if match else ""


def infer_category(text: str) -> str:
    """Infer a coarse category from common markers in TechQA documents."""

    upper_text = text.upper()
    for marker, label in CATEGORY_MARKERS.items():
        if marker in upper_text:
            return label
    return ""


def prepare_techqa_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize TechQA documents and drop exact duplicates."""

    prepared: list[dict[str, Any]] = []
    seen_checksums: set[str] = set()
    for raw in raw_documents:
        text = normalize_text(str(raw.get("text", "")))
        if not text:
            continue
        checksum = stable_checksum(text)
        if checksum in seen_checksums:
            continue
        seen_checksums.add(checksum)

        title = normalize_text(str(raw.get("title", ""))) or infer_title(text)
        product = normalize_text(str(raw.get("product", ""))) or infer_product(text)
        version = normalize_text(str(raw.get("version", ""))) or infer_version(text)
        category = normalize_label(str(raw.get("category", ""))) or infer_category(text)
        date = normalize_text(str(raw.get("date", "")))
        source_doc_id = normalize_text(str(raw.get("doc_id", "")))

        prepared.append(
            {
                "doc_id": source_doc_id or checksum[:16],
                "source_doc_id": source_doc_id,
                "title": title,
                "text": text,
                "product": product,
                "version": version,
                "category": category,
                "date": date,
                "source": "techqa",
                "checksum": checksum,
                "char_count": len(text),
                "word_count": len(text.split()),
            }
        )
    return prepared


def prepare_qa_pairs(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Normalize question-answer corpora into a shared evaluation schema."""

    prepared: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in rows:
        question = normalize_text(str(row.get("question", "")))
        answer = normalize_text(str(row.get("answer", "")))
        if not question or not answer:
            continue

        question_key = question.lower()
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)

        explicit_id = row.get("question_id") or row.get("id") or row.get("conversation_id") or ""
        generated_id = f"{source}_{stable_checksum(question)[:12]}"
        prepared.append(
            {
                "question_id": normalize_text(str(explicit_id)) or generated_id,
                "question": question,
                "answer": answer,
                "doc_id": normalize_text(str(row.get("doc_id", ""))),
                "intent": normalize_label(str(row.get("intent", ""))),
                "category": normalize_label(str(row.get("category", ""))),
                "source": source,
            }
        )
    return prepared


def prepare_msdialog_conversations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize MSDialog conversations for future agent evaluation scenarios."""

    prepared: list[dict[str, Any]] = []
    for row in rows:
        text = normalize_text(str(row.get("text", "")))
        if not text:
            continue
        conversation_id = normalize_text(str(row.get("conversation_id", "")))
        prepared.append(
            {
                "conversation_id": conversation_id or stable_checksum(text)[:12],
                "text": text,
                "final_answer": normalize_text(str(row.get("final_answer", ""))),
                "intent": normalize_label(str(row.get("intent", ""))),
                "category": normalize_label(str(row.get("category", ""))),
                "source": "msdialog",
                "checksum": stable_checksum(text),
            }
        )
    return prepared
