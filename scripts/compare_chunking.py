"""Compare fixed, recursive, and BGE-M3 semantic chunking independently."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
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
    fixed_size_chunks,
    recursive_chunks,
    semantic_chunks,
)
from helpdeskai.ingestion.exceptions import TechQAIngestionError  # noqa: E402
from helpdeskai.ingestion.io import read_jsonl  # noqa: E402


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


def run_comparison(
    processed_dir: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
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
    tokenizer = HuggingFaceTokenizer(DEFAULT_MODEL)
    embedder = BgeM3Embedder()
    benchmark = compare_strategies(
        sample,
        {
            "fixed": lambda text: fixed_size_chunks(text, tokenizer),
            "recursive": lambda text: recursive_chunks(text, tokenizer),
            "semantic": lambda text: semantic_chunks(text, tokenizer, embedder),
        },
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
            force=args.force,
        )
    except (OSError, json.JSONDecodeError, TechQAIngestionError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
