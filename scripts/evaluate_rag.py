"""Evaluate Phase 4 RAG prompt variants with Ragas."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.rag.evaluation import (  # noqa: E402
    build_ragas_rows,
    build_results_for_prompt,
    evaluate_ragas_rows,
    load_ragas_cases,
    prompt_names,
    summarize_scores,
    write_comparison,
    write_rag_results,
)
from helpdeskai.rag.llm import MissingAnthropicKeyError  # noqa: E402
from helpdeskai.rag.models import RagConfig  # noqa: E402
from helpdeskai.rag.prompts import PROMPT_VARIANTS  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-path", type=Path, default=Path("tests/golden/questions.jsonl"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/rag"))
    parser.add_argument("--prompt", default="all", choices=["all", *sorted(PROMPT_VARIANTS)])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--generator-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    return parser.parse_args(argv)


def _print_summary(summaries: dict[str, dict[str, float]]) -> None:
    table = Table(title="Ragas prompt comparison", show_lines=True)
    table.add_column("Metric", style="cyan")
    for name in summaries:
        table.add_column(name, justify="right")
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        table.add_row(metric, *[f"{summaries[name].get(metric, 0.0):.4f}" for name in summaries])
    console.print(table)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        cases = load_ragas_cases(args.golden_path, limit=args.limit)
        if not cases:
            console.print("[red]error: no retrieval-eligible TechQA golden cases found[/red]")
            return 1

        summaries = {}
        for prompt in prompt_names(args.prompt):
            console.print(f"\n[cyan]Running RAG for prompt `{prompt}` on {len(cases)} cases[/cyan]")
            config = RagConfig(
                generator_model=args.generator_model,
                judge_model=args.judge_model,
                prompt_version=prompt,
                retrieval_mode=args.mode,
            )
            results = build_results_for_prompt(cases, config=config)
            write_rag_results(args.report_dir / f"rag_results_{prompt}.jsonl", results)
            rows = build_ragas_rows(cases, results)

            console.print(f"[cyan]Evaluating prompt `{prompt}` with Ragas[/cyan]")
            dataframe = evaluate_ragas_rows(
                rows,
                judge_model=args.judge_model,
                embedding_model=args.embedding_model,
            )
            dataframe.to_csv(args.report_dir / f"ragas_results_{prompt}.csv", index=False)
            summaries[prompt] = summarize_scores(dataframe)

        write_comparison(args.report_dir, summaries)
        _print_summary(summaries)
        console.print(f"\n[green]Wrote RAG evaluation artifacts to {args.report_dir}[/green]")
    except MissingAnthropicKeyError as exc:
        console.print(f"[red]error: {exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
