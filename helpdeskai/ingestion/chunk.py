"""Selected recursive chunking stage for the production ingestion pipeline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

DEFAULT_TOKENIZER = "BAAI/bge-m3"
CHUNKING_VERSION = "recursive-1.0"
TARGET_TOKENS = 384
OVERLAP_TOKENS = 64

PARAGRAPH_RE = re.compile(r"\n\s*\n+")
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9])")
TOKEN_RE = re.compile(r"\S+")


class Tokenizer(Protocol):
    """Minimal tokenizer interface required by the selected chunker."""

    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class WhitespaceTokenizer:
    """Deterministic tokenizer used by unit tests."""

    name = "whitespace-v1"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        result = []
        for token in TOKEN_RE.findall(text):
            if token not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            result.append(self._token_to_id[token])
        return result

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self._id_to_token[index] for index in token_ids)


class HuggingFaceTokenizer:
    """Tokenizer adapter for the selected BGE-M3 retrieval model."""

    def __init__(self, model_name: str = DEFAULT_TOKENIZER) -> None:
        from transformers import AutoTokenizer

        self.name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.model_max_length = 1_000_000_000

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()


def _windows(token_ids: Sequence[int], size: int, overlap: int) -> list[Sequence[int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between 0 and chunk size")
    step = size - overlap
    return [token_ids[start : start + size] for start in range(0, len(token_ids), step)]


def _decode_bounded(
    token_ids: Sequence[int],
    tokenizer: Tokenizer,
    limit: int,
) -> tuple[str, int]:
    bounded = list(token_ids)
    content = tokenizer.decode(bounded).strip()
    count = len(tokenizer.encode(content)) if content else 0
    while content and count > limit:
        bounded.pop()
        content = tokenizer.decode(bounded).strip()
        count = len(tokenizer.encode(content)) if content else 0
    return content, count


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in PARAGRAPH_RE.split(text):
        sentences.extend(
            part.strip() for part in SENTENCE_RE.split(paragraph) if part.strip()
        )
    return sentences


def _split_oversized(text: str, tokenizer: Tokenizer, limit: int) -> list[str]:
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= limit:
        return [text.strip()]
    parts = []
    for window in _windows(token_ids, limit, 0):
        content, _ = _decode_bounded(window, tokenizer, limit)
        if content:
            parts.append(content)
    return parts


def recursive_chunks(
    text: str,
    tokenizer: Tokenizer,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Split text on paragraphs and sentences with bounded token overlap."""
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap must be between 0 and target tokens")
    paragraphs = [part.strip() for part in PARAGRAPH_RE.split(text) if part.strip()]
    units = []
    for paragraph in paragraphs:
        if len(tokenizer.encode(paragraph)) <= target_tokens:
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))

    expanded = [
        part
        for unit in units
        for part in _split_oversized(unit, tokenizer, target_tokens)
        if part
    ]
    base: list[tuple[str, int]] = []
    current: list[str] = []
    current_count = 0
    for unit in expanded:
        count = len(tokenizer.encode(unit))
        separator_tokens = 1 if current else 0
        if current and current_count + separator_tokens + count > target_tokens:
            content = "\n\n".join(current)
            base.append((content, len(tokenizer.encode(content))))
            current, current_count = [], 0
        current.append(unit)
        current_count += separator_tokens + count
    if current:
        content = "\n\n".join(current)
        base.append((content, len(tokenizer.encode(content))))

    chunks = []
    previous_tail: list[int] = []
    for position, (content, count) in enumerate(base):
        current_ids = tokenizer.encode(content)
        if position and overlap_tokens:
            combined = [*previous_tail, *current_ids]
            if len(combined) > target_tokens:
                combined = combined[-target_tokens:]
            content, count = _decode_bounded(combined, tokenizer, target_tokens)
        chunks.append(
            {
                "content": content,
                "token_count": count,
                "position": position,
            }
        )
        previous_tail = current_ids[-overlap_tokens:]
    return chunks


def chunk_documents(
    documents: Sequence[dict],
    tokenizer: Tokenizer,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Create index-ready recursive chunks with inherited provenance."""
    records = []
    parameters = {
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
    }
    for document in documents:
        for chunk in recursive_chunks(
            document["text"],
            tokenizer,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        ):
            content_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
            records.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}#{chunk['position']:04d}-"
                        f"{content_hash[:12]}"
                    ),
                    "document_id": document["document_id"],
                    "position": chunk["position"],
                    "content": chunk["content"],
                    "content_hash": content_hash,
                    "token_count": chunk["token_count"],
                    "metadata": document["metadata"],
                    "source_ids": document["source_ids"],
                    "splits": document["splits"],
                    "chunking": {
                        "version": CHUNKING_VERSION,
                        "strategy": "recursive",
                        "parameters": parameters,
                        "tokenizer": tokenizer.name,
                    },
                }
            )
    return records
