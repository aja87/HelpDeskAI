"""Run the advanced RAG pipeline with Claude generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.rag.llm import MissingAnthropicKeyError  # noqa: E402
from helpdeskai.rag.models import RagConfig  # noqa: E402
from helpdeskai.rag.pipeline import AdvancedRagPipeline  # noqa: E402
from helpdeskai.rag.prompts import PROMPT_VARIANTS  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        action="append",
        help="Question to answer. Can be passed multiple times.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="Optional UTF-8 file with one question per line.",
    )
    parser.add_argument("--prompt", choices=sorted(PROMPT_VARIANTS), default="strict")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=8_000)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    return parser.parse_args(argv)


def _questions(args: argparse.Namespace) -> list[str]:
    questions = list(args.question or [])
    if args.questions_file:
        questions.extend(
            line.strip()
            for line in args.questions_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return questions or ["How do I configure SAML authentication?"]


def _print_result(result) -> None:
    console.print(f"\n[bold cyan]Question[/bold cyan] {result.question_original}")
    if result.question_rewritten != result.question_original:
        console.print(f"[orange3]Rewritten[/orange3] {result.question_rewritten}")

    console.print("\n[bold]Top contexts[/bold]")
    for context in result.contexts:
        console.print(
            f"  [{context.chunk_id}] doc={context.document_id} score={context.score:.4f}"
        )
    console.print(Panel(result.answer, title="Answer", border_style="orange3"))

    table = Table(title="Stage timings")
    table.add_column("Stage", style="cyan")
    table.add_column("Latency ms", justify="right")
    for timing in result.timings:
        table.add_row(timing.name, f"{timing.duration_ms:.0f}")
    console.print(table)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        pipeline = AdvancedRagPipeline(
            config=RagConfig(
                generator_model=args.model,
                retrieval_mode=args.mode,
                prompt_version=args.prompt,
                candidate_k=args.candidate_k,
                final_k=args.final_k,
                max_context_chars=args.max_context_chars,
            )
        )
        for question in _questions(args):
            _print_result(pipeline.run(question))
    except MissingAnthropicKeyError as exc:
        console.print(f"[red]error: {exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
