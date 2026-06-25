"""Standalone chunking strategies used to prepare documents for retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"
CHUNKING_VERSION = "1.0"

PARAGRAPH_RE = re.compile(r"\n\s*\n+")
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9])")
TOKEN_RE = re.compile(r"\S+")


class Tokenizer(Protocol):
    """Minimal tokenizer contract required by the chunkers."""

    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class Embedder(Protocol):
    """Minimal normalized embedding contract."""

    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class WhitespaceTokenizer:
    """Deterministic lightweight tokenizer used by tests and fallbacks."""

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
    """Tokenizer adapter loaded from a pinned Hugging Face model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from transformers import AutoTokenizer

        self.name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Full documents are counted before splitting but are never embedded
        # as a single sequence.
        self._tokenizer.model_max_length = 1_000_000_000

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()


class BgeM3Embedder:
    """Local BGE-M3 sentence embedder with CPU-safe defaults."""

    model_name = DEFAULT_MODEL

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int | None = None,
        max_seq_length: int = 256,
        cache_folder: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": "float16"} if device.startswith("cuda") else {}
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
        )
        self._model.max_seq_length = max_seq_length
        self.batch_size = batch_size or (16 if device.startswith("cuda") else 4)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )


@dataclass(frozen=True)
class Chunk:
    """A chunk before document metadata is attached."""

    content: str
    token_count: int
    position: int
    strategy: str


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


def fixed_size_chunks(
    text: str,
    tokenizer: Tokenizer,
    *,
    target_tokens: int = 384,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split text into fixed token windows."""
    token_ids = tokenizer.encode(text)
    chunks = []
    for position, window in enumerate(_windows(token_ids, target_tokens, overlap_tokens)):
        content, count = _decode_bounded(window, tokenizer, target_tokens)
        if content:
            chunks.append(Chunk(content, count, position, "fixed"))
    return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in PARAGRAPH_RE.split(text):
        sentences.extend(part.strip() for part in SENTENCE_RE.split(paragraph) if part.strip())
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


def _pack_units(
    units: Sequence[str],
    tokenizer: Tokenizer,
    *,
    target_tokens: int,
    strategy: str,
) -> list[Chunk]:
    expanded = [
        part for unit in units for part in _split_oversized(unit, tokenizer, target_tokens) if part
    ]
    packed: list[Chunk] = []
    current: list[str] = []
    current_count = 0
    for unit in expanded:
        count = len(tokenizer.encode(unit))
        separator_tokens = 1 if current else 0
        if current and current_count + separator_tokens + count > target_tokens:
            content = "\n\n".join(current)
            packed.append(Chunk(content, len(tokenizer.encode(content)), len(packed), strategy))
            current = []
            current_count = 0
        current.append(unit)
        current_count += separator_tokens + count
    if current:
        content = "\n\n".join(current)
        packed.append(Chunk(content, len(tokenizer.encode(content)), len(packed), strategy))
    return packed


def recursive_chunks(
    text: str,
    tokenizer: Tokenizer,
    *,
    target_tokens: int = 384,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split on paragraphs, then sentences, retaining a bounded textual overlap."""
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap must be between 0 and target tokens")
    paragraphs = [part.strip() for part in PARAGRAPH_RE.split(text) if part.strip()]
    units = []
    for paragraph in paragraphs:
        if len(tokenizer.encode(paragraph)) <= target_tokens:
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))
    base = _pack_units(units, tokenizer, target_tokens=target_tokens, strategy="recursive")
    if overlap_tokens == 0 or len(base) < 2:
        return base

    overlapped = []
    previous_tail: list[int] = []
    for position, chunk in enumerate(base):
        current_ids = tokenizer.encode(chunk.content)
        combined = [*previous_tail, *current_ids]
        if len(combined) > target_tokens:
            combined = combined[-target_tokens:]
        content, count = _decode_bounded(combined, tokenizer, target_tokens)
        overlapped.append(Chunk(content, count, position, "recursive"))
        previous_tail = current_ids[-overlap_tokens:]
    return overlapped


def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.empty(0, dtype=np.float32)
    return 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)


def semantic_chunks(
    text: str,
    tokenizer: Tokenizer,
    embedder: Embedder,
    *,
    min_tokens: int = 128,
    max_tokens: int = 512,
    breakpoint_percentile: float = 80.0,
) -> list[Chunk]:
    """Split at large adjacent-sentence embedding changes."""
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError("breakpoint percentile must be between 0 and 100")
    sentences = _split_sentences(text)
    if not sentences:
        return []
    vectors = embedder.encode(sentences)
    distances = _cosine_distances(vectors)
    threshold = (
        float(np.percentile(distances, breakpoint_percentile)) if len(distances) else math.inf
    )

    groups: list[list[str]] = [[]]
    group_tokens = 0
    for index, sentence in enumerate(sentences):
        sentence_tokens = len(tokenizer.encode(sentence))
        boundary = index > 0 and distances[index - 1] >= threshold and group_tokens >= min_tokens
        overflow = groups[-1] and group_tokens + sentence_tokens > max_tokens
        if boundary or overflow:
            groups.append([])
            group_tokens = 0
        groups[-1].append(sentence)
        group_tokens += sentence_tokens

    chunks = []
    for group in groups:
        content = " ".join(group).strip()
        for part in _split_oversized(content, tokenizer, max_tokens):
            chunks.append(
                Chunk(
                    content=part,
                    token_count=len(tokenizer.encode(part)),
                    position=len(chunks),
                    strategy="semantic",
                )
            )
    return chunks


Chunker = Callable[[str], list[Chunk]]


def stable_chunk_id(document_id: str, position: int, content: str) -> str:
    """Create a stable ID that changes when chunk content changes."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}#{position:04d}-{digest}"


def build_chunk_records(
    documents: Sequence[dict],
    chunker: Chunker,
    *,
    tokenizer_name: str,
    strategy: str,
    strategy_params: dict,
    embedding_model: str | None = None,
) -> list[dict]:
    """Attach provenance and inherited metadata to document chunks."""
    records = []
    for document in documents:
        document_id = document["document_id"]
        for chunk in chunker(document["text"]):
            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            records.append(
                {
                    "chunk_id": stable_chunk_id(document_id, chunk.position, chunk.content),
                    "document_id": document_id,
                    "position": chunk.position,
                    "content": chunk.content,
                    "content_hash": content_hash,
                    "token_count": chunk.token_count,
                    "metadata": document["metadata"],
                    "source_ids": document["source_ids"],
                    "splits": document["splits"],
                    "chunking": {
                        "version": CHUNKING_VERSION,
                        "strategy": strategy,
                        "parameters": strategy_params,
                        "tokenizer": tokenizer_name,
                        "embedding_model": embedding_model,
                    },
                }
            )
    return records
