"""Register the complete HelpDeskAI RAG chain as an MLflow pyfunc model."""

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

from helpdeskai.observability.mlflow_model import (  # noqa: E402
    DEFAULT_RAG_MODEL_EXPERIMENT,
    DEFAULT_RAG_MODEL_NAME,
    register_rag_pyfunc_model,
)
from helpdeskai.rag.models import RagConfig  # noqa: E402
from helpdeskai.rag.prompts import PROMPT_VARIANTS  # noqa: E402
from helpdeskai.retrieval.models import SearchMode  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default=DEFAULT_RAG_MODEL_EXPERIMENT)
    parser.add_argument("--run-name", default="register_rag_chain")
    parser.add_argument("--model-name", default=DEFAULT_RAG_MODEL_NAME)
    parser.add_argument("--artifact-path", default="rag_chain")
    parser.add_argument("--alias", default="production")
    parser.add_argument("--generator-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--mode", choices=[mode.value for mode in SearchMode], default="hybrid")
    parser.add_argument("--prompt", choices=sorted(PROMPT_VARIANTS), default="strict")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=8_000)
    parser.add_argument("--max-generation-tokens", type=int, default=500)
    parser.add_argument("--max-rewrite-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-code-paths",
        action="store_true",
        help="Do not package the local helpdeskai source tree into the MLflow artifact.",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> RagConfig:
    return RagConfig(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        reranker_model=args.reranker_model,
        retrieval_mode=args.mode,
        prompt_version=args.prompt,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        max_context_chars=args.max_context_chars,
        max_generation_tokens=args.max_generation_tokens,
        max_rewrite_tokens=args.max_rewrite_tokens,
        temperature=args.temperature,
    )


def _print_result(result: dict[str, str | None]) -> None:
    table = Table(title="Registered RAG pyfunc model")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key in ["model_name", "model_version", "alias", "model_uri", "run_id"]:
        table.add_row(key, str(result.get(key) or "n/a"))
    console.print(table)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    code_paths = None if args.skip_code_paths else [PROJECT_ROOT / "helpdeskai"]
    result = register_rag_pyfunc_model(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        run_name=args.run_name,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        config=_config_from_args(args),
        code_paths=code_paths,
        alias=args.alias or None,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
