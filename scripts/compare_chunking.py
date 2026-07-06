"""Compare fixed, recursive, and semantic chunking independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.corpus.benchmark import (  # noqa: E402
    compare_strategies,
    deterministic_document_sample,
    write_benchmark,
)
from helpdeskai.corpus.chunking import (  # noqa: E402
    DEFAULT_MODEL,
    BgeM3Embedder,
    HuggingFaceTokenizer,
    WhitespaceTokenizer,
    fixed_size_chunks,
    recursive_chunks,
    semantic_chunks,
)
from helpdeskai.ingestion.exceptions import TechQAIngestionError  # noqa: E402
from helpdeskai.ingestion.io import read_jsonl  # noqa: E402

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class HashingSemanticEmbedder:
    """Low-memory deterministic sentence embedder for chunking comparisons."""

    model_name = "hashing-semantic-fallback-v1"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                column = value % self.dimensions
                sign = 1.0 if value & 1 else -1.0
                vectors[row, column] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/techqa"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/corpus_preparation"),
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        choices=("bge-m3", "whitespace"),
        default="bge-m3",
        help="Tokenizer used for the comparison. Use 'whitespace' on low-memory machines.",
    )
    parser.add_argument(
        "--semantic-embedder",
        choices=("auto", "bge-m3", "hashing"),
        default="auto",
        help=(
            "Semantic chunking embedder. 'auto' tries BGE-M3 and falls back to a "
            "low-memory hashing embedder if the model cannot be loaded."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_benchmark(path: Path) -> pd.DataFrame:
    """Load strategy metrics from a benchmark JSON artifact."""
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    strategies = benchmark.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("benchmark does not contain strategy metrics")
    return pd.DataFrame.from_dict(strategies, orient="index")


def write_comparison(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Write a multi-panel strategy comparison chart."""
    required = ("mean_tokens", "median_tokens", "runtime_seconds")
    missing = [column for column in required if column not in dataframe]
    if missing:
        raise ValueError(f"benchmark is missing columns: {', '.join(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axes = dataframe[list(required)].plot(
        kind="bar",
        subplots=True,
        figsize=(10, 10),
        title="Chunking strategy comparison",
        legend=False,
    )
    axes[0].figure.tight_layout()
    axes[0].figure.savefig(output_path, dpi=150)
    plt.close(axes[0].figure)
    return output_path


def build_semantic_embedder(kind: str):
    """Build the semantic chunking embedder with an explicit low-memory fallback."""
    if kind == "hashing":
        return HashingSemanticEmbedder()

    try:
        return BgeM3Embedder()
    except Exception as exc:
        if kind == "bge-m3":
            raise RuntimeError(
                "BGE-M3 semantic embedder could not be loaded. Use "
                "--semantic-embedder hashing for the low-memory comparison."
            ) from exc
        print(
            "warning: BGE-M3 semantic embedder unavailable; using "
            f"{HashingSemanticEmbedder.model_name}.",
            file=sys.stderr,
        )
        return HashingSemanticEmbedder()


def build_tokenizer(kind: str):
    """Build the tokenizer used by all chunking strategies."""
    if kind == "whitespace":
        return WhitespaceTokenizer()
    return HuggingFaceTokenizer(DEFAULT_MODEL)


def run_comparison(
    processed_dir: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
    tokenizer: str,
    semantic_embedder: str,
    force: bool,
) -> tuple[Path, Path, Path]:
    """Execute the full comparison and write JSON, Markdown, and PNG outputs."""
    expected = (
        output_dir / "chunking_benchmark.json",
        output_dir / "chunking_benchmark.md",
        output_dir / "chunking_comparison.png",
    )
    if any(path.exists() for path in expected) and not force:
        raise FileExistsError("chunking comparison exists; use --force to replace it")

    documents = read_jsonl(processed_dir / "documents.jsonl")
    sample = deterministic_document_sample(
        documents,
        sample_size=sample_size,
        seed=seed,
    )
    tokenizer_instance = build_tokenizer(tokenizer)
    embedder = build_semantic_embedder(semantic_embedder)
    benchmark = compare_strategies(
        sample,
        {
            "fixed": lambda text: fixed_size_chunks(text, tokenizer_instance),
            "recursive": lambda text: recursive_chunks(text, tokenizer_instance),
            "semantic": lambda text: semantic_chunks(text, tokenizer_instance, embedder),
        },
    )
    benchmark["sample"] = {"documents": len(sample), "seed": seed}
    benchmark["tokenizer"] = getattr(tokenizer_instance, "name", tokenizer)
    benchmark["semantic_embedder"] = getattr(embedder, "model_name", semantic_embedder)
    benchmark["selected_strategy"] = "recursive"
    benchmark["justification"] = (
        "Recursive chunking is retained for production ingestion because it preserves "
        "paragraph and sentence boundaries, keeps chunk sizes bounded without requiring "
        "an embedding model at ingestion time, and is faster/more robust than semantic "
        "chunking on local demo hardware. Fixed-size chunking is simpler but can cut "
        "semantic units in the middle of a paragraph."
    )
    json_path, markdown_path = write_benchmark(output_dir, benchmark)
    chart_path = write_comparison(
        pd.DataFrame.from_dict(benchmark["strategies"], orient="index"),
        output_dir / "chunking_comparison.png",
    )
    return json_path, markdown_path, chart_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = run_comparison(
            args.processed_dir,
            args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            tokenizer=args.tokenizer,
            semantic_embedder=args.semantic_embedder,
            force=args.force,
        )
    except (
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        TechQAIngestionError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
