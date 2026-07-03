"""Chunking strategies and benchmark helpers for ingestion."""

from __future__ import annotations

import random
import re
from statistics import mean, median
from typing import Any, Callable

from .config import IngestionConfig
from .normalize import normalize_text


def _with_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Apply simple character overlap between already packed chunks."""

    if overlap <= 0:
        return chunks
    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue
        prefix = chunks[index - 1][-overlap:].strip()
        merged = f"{prefix}\n{chunk}" if prefix else chunk
        overlapped.append(merged.strip())
    return overlapped


def chunk_fixed_size(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[str]:
    """Chunk text by fixed character windows."""

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _pack_segments(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Pack semantic segments into chunks close to a target size."""

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        segment_size = len(segment)
        if current and current_size + segment_size + 2 > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = [segment]
            current_size = segment_size
            continue
        current.append(segment)
        current_size += segment_size + 2

    if current:
        chunks.append("\n\n".join(current).strip())
    return _with_overlap(chunks, overlap)


def chunk_recursive(text: str, chunk_size: int = 1000, overlap: int = 120) -> list[str]:
    """Chunk by paragraphs first, then by sentence groups for long sections."""

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    expanded_segments: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            expanded_segments.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        expanded_segments.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return _pack_segments(expanded_segments, chunk_size, overlap)


def chunk_semantic(text: str, chunk_size: int = 900) -> list[str]:
    """Approximate semantic chunking by favoring section boundaries and headings."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in lines:
        is_heading = line.isupper() or line.endswith(":") or len(line) < 80
        line_size = len(line)
        if current and (is_heading or current_size + line_size + 1 > chunk_size):
            segments.append("\n".join(current).strip())
            current = [line]
            current_size = line_size
            continue
        current.append(line)
        current_size += line_size + 1

    if current:
        segments.append("\n".join(current).strip())
    return segments


def benchmark_chunking_strategies(
    documents: list[dict[str, Any]], config: IngestionConfig
) -> dict[str, Any]:
    """Compare fixed, recursive, and semantic chunking on a deterministic sample."""

    rng = random.Random(config.seed)
    sample_size = min(config.chunk_sample_size, len(documents))
    sampled_documents = rng.sample(documents, sample_size) if sample_size else []

    strategies: dict[str, Callable[[str], list[str]]] = {
        "fixed": lambda text: chunk_fixed_size(text, config.fixed_chunk_size, config.fixed_overlap),
        "recursive": lambda text: chunk_recursive(
            text, config.recursive_chunk_size, config.recursive_overlap
        ),
        "semantic": lambda text: chunk_semantic(text, config.semantic_chunk_size),
    }

    results: dict[str, Any] = {}
    target = config.recursive_chunk_size
    for name, chunker in strategies.items():
        chunk_lengths: list[int] = []
        chunks_per_doc: list[int] = []
        for document in sampled_documents:
            chunks = [chunk for chunk in chunker(document["text"]) if chunk.strip()]
            if not chunks:
                continue
            chunks_per_doc.append(len(chunks))
            chunk_lengths.extend(len(chunk) for chunk in chunks)

        if not chunk_lengths:
            results[name] = {
                "chunk_count": 0,
                "avg_chunk_chars": 0,
                "median_chunk_chars": 0,
                "avg_chunks_per_doc": 0,
                "oversized_ratio": 0,
                "undersized_ratio": 0,
                "score": 9999,
            }
            continue

        oversized_ratio = sum(length > 1400 for length in chunk_lengths) / len(chunk_lengths)
        undersized_ratio = sum(length < 300 for length in chunk_lengths) / len(chunk_lengths)
        avg_chunk_chars = mean(chunk_lengths)
        score = abs(avg_chunk_chars - target) + 150 * oversized_ratio + 75 * undersized_ratio

        results[name] = {
            "chunk_count": len(chunk_lengths),
            "avg_chunk_chars": round(avg_chunk_chars, 2),
            "median_chunk_chars": round(median(chunk_lengths), 2),
            "avg_chunks_per_doc": round(mean(chunks_per_doc), 2),
            "oversized_ratio": round(oversized_ratio, 4),
            "undersized_ratio": round(undersized_ratio, 4),
            "score": round(score, 2),
        }

    recommended = min(results.items(), key=lambda item: item[1]["score"])[0] if results else "recursive"
    return {
        "sample_size": sample_size,
        "recommended_strategy": recommended,
        "strategies": results,
    }


def chunk_documents(
    documents: list[dict[str, Any]], strategy_name: str, config: IngestionConfig
) -> list[dict[str, Any]]:
    """Create chunk-level records ready for vector indexing."""

    strategies: dict[str, Callable[[str], list[str]]] = {
        "fixed": lambda text: chunk_fixed_size(text, config.fixed_chunk_size, config.fixed_overlap),
        "recursive": lambda text: chunk_recursive(
            text, config.recursive_chunk_size, config.recursive_overlap
        ),
        "semantic": lambda text: chunk_semantic(text, config.semantic_chunk_size),
    }
    chunker = strategies[strategy_name]

    chunk_rows: list[dict[str, Any]] = []
    for document in documents:
        for index, chunk in enumerate(chunker(document["text"])):
            normalized_chunk = normalize_text(chunk)
            if not normalized_chunk:
                continue
            chunk_rows.append(
                {
                    "chunk_id": f"{document['doc_id']}_{index:04d}",
                    "doc_id": document["doc_id"],
                    "chunk_index": index,
                    "strategy": strategy_name,
                    "text": normalized_chunk,
                    "title": document["title"],
                    "product": document["product"],
                    "version": document["version"],
                    "category": document["category"],
                    "date": document["date"],
                    "source": document["source"],
                }
            )
    return chunk_rows
