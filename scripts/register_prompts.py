"""Register HelpDeskAI RAG prompts in MLflow and manage aliases."""

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

from helpdeskai.observability.prompt_registry import (  # noqa: E402
    PROMPT_EXPERIMENT,
    PROMPT_NAME,
    load_prompt_by_alias,
    promote_prompt_alias,
    register_prompt_versions,
)
from helpdeskai.rag.prompts import PROMPT_VARIANTS  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default=PROMPT_EXPERIMENT)
    parser.add_argument("--prompt-name", default=PROMPT_NAME)
    parser.add_argument(
        "--promote-from-eval",
        action="store_true",
        help="Promote the best prompt from MLflow evaluation runs to production.",
    )
    parser.add_argument("--eval-experiment", default="helpdeskai-rag-eval")
    return parser.parse_args(argv)


def _prompt_texts() -> dict[str, str]:
    return {
        name: template("{question}", "{context}")
        for name, template in PROMPT_VARIANTS.items()
    }


def _best_prompt(eval_experiment: str) -> str | None:
    runs = mlflow.search_runs(experiment_names=[eval_experiment])
    if runs.empty:
        return None
    candidates = []
    for prompt in PROMPT_VARIANTS:
        column = f"metrics.{prompt}.faithfulness"
        if column in runs.columns:
            best = runs.loc[runs[column].idxmax()]
            candidates.append((prompt, float(best[column])))
    if candidates:
        return max(candidates, key=lambda item: item[1])[0]
    if "params.prompt_version" in runs.columns and "metrics.faithfulness" in runs.columns:
        best_row = runs.loc[runs["metrics.faithfulness"].idxmax()]
        return str(best_row["params.prompt_version"])
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mlflow.set_tracking_uri(args.tracking_uri)
    run_ids = register_prompt_versions(
        _prompt_texts(),
        aliases={"dev": "concise", "staging": "pedagogical", "production": "strict"},
        prompt_name=args.prompt_name,
        experiment=args.experiment,
    )

    if args.promote_from_eval:
        winner = _best_prompt(args.eval_experiment)
        if winner in run_ids:
            promote_prompt_alias(
                args.prompt_name,
                winner,
                "production",
                experiment=args.experiment,
            )
            console.print(f"[green]Promoted {winner} to production[/green]")
        else:
            console.print("[yellow]No evaluation winner found; kept default aliases[/yellow]")

    table = Table(title="Prompt registry aliases")
    table.add_column("Alias", style="cyan")
    table.add_column("Version")
    table.add_column("Preview", overflow="fold")
    for alias in ["dev", "staging", "production"]:
        loaded = load_prompt_by_alias(args.prompt_name, alias, experiment=args.experiment)
        if loaded:
            version, text = loaded
            table.add_row(alias, version, text.replace("\n", " ")[:100])
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
