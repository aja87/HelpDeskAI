"""Record mapping helpers for raw corpus datasets."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def as_text(value: Any) -> str:
    """Normalize heterogeneous dataset values into plain text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def pick_first(row: dict[str, Any], candidates: list[str]) -> Any:
    """Return the first non-empty value found for candidate keys."""

    for key in candidates:
        if key in row and row[key] not in (None, "", []):
            return row[key]
    return None


def map_techqa_doc(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw TechQA document row to the normalized output schema."""

    text = as_text(
        pick_first(
            row,
            [
                "document",
                "doc",
                "text",
                "content",
                "html",
                "body",
                "passage",
            ],
        )
    ).strip()
    if not text:
        return None
    doc_id = as_text(pick_first(row, ["id", "doc_id", "document_id", "uid"]))
    if not doc_id:
        # Generate a stable fallback id when source id is missing.
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    return {
        "doc_id": doc_id,
        "text": text,
        "title": as_text(pick_first(row, ["title", "subject"])),
        "product": as_text(pick_first(row, ["product", "product_name"])),
        "version": as_text(pick_first(row, ["version", "product_version"])),
        "category": as_text(pick_first(row, ["category", "topic", "type"])),
        "date": as_text(pick_first(row, ["date", "published", "updated_at"])),
        "source": "techqa",
    }


def map_techqa_qa(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw TechQA QA row to a compact evaluation-friendly schema."""

    question = as_text(pick_first(row, ["question", "query", "title"])).strip()
    answer = as_text(pick_first(row, ["answer", "answers", "response", "label"])).strip()
    if not question or not answer:
        return None
    return {
        "question_id": as_text(pick_first(row, ["qid", "id", "question_id"])),
        "question": question,
        "answer": answer,
        "doc_id": as_text(pick_first(row, ["doc_id", "document_id", "docno"])),
        "source": "techqa",
    }


def map_bitext(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw Bitext row to a unified question/answer schema."""

    question = as_text(
        pick_first(
            row,
            [
                "instruction",
                "question",
                "utterance",
                "input",
                "customer_question",
                "text",
            ],
        )
    ).strip()
    answer = as_text(
        pick_first(row, ["response", "answer", "output", "agent_response", "label"])
    ).strip()
    if not question or not answer:
        return None
    return {
        "id": as_text(pick_first(row, ["id", "uid"])),
        "question": question,
        "answer": answer,
        "intent": as_text(pick_first(row, ["intent", "intent_label"])),
        "category": as_text(pick_first(row, ["category", "topic"])),
        "source": "bitext",
    }


def map_msdialog(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw MSDialog row to a normalized conversation schema."""

    conversation = pick_first(
        row,
        [
            "conversations",
            "utterances",
            "dialog",
            "messages",
            "conversation",
            "turns",
            "text",
        ],
    )
    text = as_text(conversation).strip()
    if not text:
        return None
    return {
        "conversation_id": as_text(pick_first(row, ["id", "conversation_id", "uid"])),
        "text": text,
        "final_answer": as_text(pick_first(row, ["final_answer", "answer", "response"])),
        "intent": as_text(pick_first(row, ["intent", "dialog_act", "label"])),
        "category": as_text(pick_first(row, ["category", "topic", "domain"])),
        "source": "msdialog",
    }
