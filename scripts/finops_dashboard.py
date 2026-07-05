"""Compute HelpDeskAI FinOps scenarios and export reports."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.observability.finops import (  # noqa: E402
    Scenario,
    default_scenarios,
    make_optimized,
    recommend,
)

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("reports/finops/scenarios.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/finops/summary.md"))
    return parser.parse_args(argv)


def scenario_rows(scenarios: list[Scenario]) -> list[dict]:
    rows = []
    for scenario in scenarios:
        rows.append(scenario.to_row(variant="baseline"))
        rows.append(make_optimized(scenario).to_row(variant="optimized"))
    return rows


def export_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_markdown(scenarios: list[Scenario], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HelpDeskAI FinOps summary",
        "",
        "| Scenario | Baseline/month | Optimized/month | Savings | Optimized cost/query |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        baseline = scenario.effective_cost()
        optimized = make_optimized(scenario).effective_cost()
        savings = float(baseline["total_usd"]) - float(optimized["total_usd"])
        lines.append(
            f"| {scenario.name} | ${baseline['total_usd']:.2f} | "
            f"${optimized['total_usd']:.2f} | ${savings:.2f} | "
            f"${optimized['cost_per_query_usd']:.5f} |"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in recommend(scenarios))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = default_scenarios()
    table = Table(title="FinOps baseline vs optimized", show_lines=True)
    table.add_column("Scenario", style="cyan")
    table.add_column("Baseline/month", justify="right")
    table.add_column("Optimized/month", justify="right")
    table.add_column("Savings", justify="right")
    table.add_column("Cost/query", justify="right")
    for scenario in scenarios:
        baseline = scenario.effective_cost()
        optimized = make_optimized(scenario).effective_cost()
        savings = float(baseline["total_usd"]) - float(optimized["total_usd"])
        table.add_row(
            scenario.name,
            f"${baseline['total_usd']:.2f}",
            f"${optimized['total_usd']:.2f}",
            f"${savings:.2f}",
            f"${optimized['cost_per_query_usd']:.5f}",
        )
    console.print(table)
    export_csv(scenario_rows(scenarios), args.csv)
    export_markdown(scenarios, args.markdown)
    console.print(f"[green]Wrote {args.csv} and {args.markdown}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
