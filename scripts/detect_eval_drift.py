"""Detect evaluation drift from MLflow continuous-eval runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import mlflow
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.observability.continuous_eval import detect_drift  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="helpdeskai-continuous-eval")
    parser.add_argument("--baseline-size", type=int, default=7)
    parser.add_argument("--recent-size", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mlflow.set_tracking_uri(args.tracking_uri)
    result = detect_drift(
        args.experiment,
        baseline_size=args.baseline_size,
        recent_size=args.recent_size,
        threshold=args.threshold,
    )
    if result.get("error"):
        console.print(f"[yellow]{result['error']}[/yellow]")
        return 1
    table = Table(title="Faithfulness drift")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in ["baseline_mean", "recent_mean", "delta", "threshold"]:
        table.add_row(key, str(result[key]))
    console.print(table)
    if result["alert"]:
        console.print("[red]Faithfulness drift detected[/red]")
        return 2
    console.print("[green]No drift detected[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
