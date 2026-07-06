"""Run the complete corpus-preparation flow."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.ingestion.exceptions import TechQAIngestionError  # noqa: E402
from helpdeskai.ingestion.pipeline import corpus_preparation_flow  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/techqa"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/corpus_preparation/corpus_quality_report.html"),
    )
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--force", action="store_true")
    behavior.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = corpus_preparation_flow(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            report_path=args.report_path,
            force=args.force,
            skip_existing=args.skip_existing,
        )
    except (
        TechQAIngestionError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 1
    print(f"Corpus preparation complete. Quality report: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
