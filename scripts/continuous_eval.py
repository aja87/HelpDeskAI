"""Sample production-like conversations and log continuous evaluation to MLflow."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import mlflow
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.observability.continuous_eval import (  # noqa: E402
    judge_answer,
    load_conversation_samples,
)

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversations",
        type=Path,
        default=Path("reports/observability/production_conversations.jsonl"),
    )
    parser.add_argument("--sample-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="helpdeskai-continuous-eval")
    parser.add_argument("--online-judge", action="store_true")
    return parser.parse_args(argv)


def _judge_client(enabled: bool):
    if not enabled:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for --online-judge")
    from anthropic import Anthropic

    return Anthropic()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.conversations.exists():
        console.print(f"[red]error: missing conversations file {args.conversations}[/red]")
        return 1

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    judge = _judge_client(args.online_judge)
    samples = load_conversation_samples(args.conversations, args.sample_ratio, args.seed)
    table = Table(title="Continuous evaluation samples", show_lines=True)
    table.add_column("Conversation")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevancy", justify="right")
    table.add_column("Reason", overflow="fold")

    for index, sample in enumerate(samples, start=1):
        scores = judge_answer(
            question=str(sample.get("question", "")),
            answer=str(sample.get("answer", "")),
            context=str(sample.get("context", "")),
            judge=judge,
        )
        conversation_id = str(sample.get("conversation_id", f"sample_{index}"))
        with mlflow.start_run(run_name=f"continuous_eval_{conversation_id}"):
            mlflow.log_param("conversation_id", conversation_id)
            mlflow.log_param("sampled", True)
            mlflow.log_metric("faithfulness", float(scores["faithfulness"]))
            mlflow.log_metric("relevancy", float(scores["relevancy"]))
            mlflow.log_text(str(scores.get("reason", "")), "judge_reason.txt")
        table.add_row(
            conversation_id,
            f"{float(scores['faithfulness']):.2f}",
            f"{float(scores['relevancy']):.2f}",
            str(scores.get("reason", ""))[:80],
        )
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
