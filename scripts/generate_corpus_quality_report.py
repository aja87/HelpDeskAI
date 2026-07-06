"""Generate the corpus quality report from existing processed chunks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.ingestion.io import read_jsonl  # noqa: E402
from helpdeskai.ingestion.quality import (  # noqa: E402
    CorpusQualityError,
    generate_evidently_report,
    validate_chunks,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/techqa"),
        help="Directory containing existing documents.jsonl and chunks.jsonl.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/corpus_preparation/corpus_quality_report.html"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("docs/corpus_preparation/corpus_quality_summary.json"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional deterministic sample size for the Evidently HTML report only.",
    )
    return parser.parse_args(argv)


def _report_chunks(chunks: list[dict], sample_size: int) -> list[dict]:
    if sample_size <= 0 or sample_size >= len(chunks):
        return chunks
    step = max(1, len(chunks) // sample_size)
    return chunks[::step][:sample_size]


def generate_report(
    *,
    processed_dir: Path,
    report_path: Path,
    summary_path: Path,
    sample_size: int = 0,
) -> tuple[Path, Path]:
    """Generate quality artifacts without modifying processed corpus files."""
    documents = read_jsonl(processed_dir / "documents.jsonl")
    chunks = read_jsonl(processed_dir / "chunks.jsonl")
    document_ids = {str(document["document_id"]) for document in documents}

    summary = validate_chunks(chunks, document_ids)
    selected_chunks = _report_chunks(chunks, sample_size)
    summary["evidently_report_chunks"] = len(selected_chunks)
    summary["validated_documents"] = len(document_ids)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    generate_evidently_report(selected_chunks, report_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, summary_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report_path, summary_path = generate_report(
            processed_dir=args.processed_dir,
            report_path=args.report_path,
            summary_path=args.summary_path,
            sample_size=args.sample_size,
        )
    except (OSError, CorpusQualityError, KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(report_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
