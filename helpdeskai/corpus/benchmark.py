"""Deterministic comparison of corpus chunking strategies."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from helpdeskai.corpus.chunking import Chunk


def deterministic_document_sample(
    documents: Sequence[dict],
    *,
    sample_size: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Sample evenly across short, medium, and long documents."""
    if sample_size > len(documents):
        raise ValueError("sample size exceeds document count")
    ordered = sorted(documents, key=lambda record: len(record["text"]))
    buckets = [ordered[index::3] for index in range(3)]
    base, remainder = divmod(sample_size, len(buckets))
    rng = random.Random(seed)
    sampled = []
    for index, bucket in enumerate(buckets):
        count = base + int(index < remainder)
        sampled.extend(rng.sample(bucket, count))
    return sorted(sampled, key=lambda record: record["document_id"])


def compare_strategies(
    documents: Sequence[dict],
    strategies: Mapping[str, Callable[[str], list[Chunk]]],
) -> dict:
    """Run every strategy over the same documents and calculate comparable metrics."""
    results = {}
    for name, chunker in strategies.items():
        started = time.perf_counter()
        chunks_by_document = {
            document["document_id"]: chunker(document["text"]) for document in documents
        }
        elapsed = time.perf_counter() - started
        chunks = [chunk for values in chunks_by_document.values() for chunk in values]
        lengths = [chunk.token_count for chunk in chunks]
        hashes = [hashlib.sha256(chunk.content.encode()).hexdigest() for chunk in chunks]
        results[name] = {
            "documents": len(documents),
            "chunks": len(chunks),
            "mean_tokens": round(statistics.fmean(lengths), 2) if lengths else 0,
            "median_tokens": round(statistics.median(lengths), 2) if lengths else 0,
            "min_tokens": min(lengths, default=0),
            "max_tokens": max(lengths, default=0),
            "undersized_chunks": sum(length < 50 for length in lengths),
            "oversized_chunks": sum(length > 512 for length in lengths),
            "duplicate_chunks": sum(count - 1 for count in Counter(hashes).values()),
            "runtime_seconds": round(elapsed, 4),
        }
    return {"sample_document_ids": [d["document_id"] for d in documents], "strategies": results}


def write_benchmark(path: Path, benchmark: dict) -> tuple[Path, Path]:
    """Write machine-readable and human-readable benchmark artifacts."""
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "chunking_benchmark.json"
    markdown_path = path / "chunking_benchmark.md"
    json_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample = benchmark.get("sample") or {}
    sample_size = sample.get("documents", len(benchmark.get("sample_document_ids", [])))
    seed = sample.get("seed", 42)
    lines = [
        "# Chunking benchmark",
        "",
        f"Deterministic comparison on {sample_size} TechQA documents (seed {seed}).",
        "",
    ]
    if benchmark.get("semantic_embedder"):
        lines.extend(["Semantic embedder: " + str(benchmark["semantic_embedder"]), ""])
    if benchmark.get("tokenizer"):
        lines.extend(["Tokenizer: " + str(benchmark["tokenizer"]), ""])
    if benchmark.get("selected_strategy"):
        lines.extend(["Selected strategy: " + str(benchmark["selected_strategy"]), ""])
    if benchmark.get("justification"):
        lines.extend(["Justification: " + str(benchmark["justification"]), ""])

    lines.extend(
        [
            "| Strategy | Chunks | Mean tokens | Median | Min | Max | Duplicates | "
            "Runtime (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in benchmark["strategies"].items():
        lines.append(
            f"| {name} | {metrics['chunks']} | {metrics['mean_tokens']} | "
            f"{metrics['median_tokens']} | {metrics['min_tokens']} | "
            f"{metrics['max_tokens']} | {metrics['duplicate_chunks']} | "
            f"{metrics['runtime_seconds']} |"
        )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
