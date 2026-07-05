
from __future__ import annotations

from argparse import Namespace, ArgumentParser, BooleanOptionalAction
import logging
import os

from pathlib import Path

from helpdeskai.common.logging import init_logging
from helpdeskai.rag.config import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EVALUATION_PATH,
    DEFAULT_GOLDEN_PATH,
    LOG_FILE,
    RagConfig,
    VALID_PROMPT_VARIANTS,
    VALID_RETRIEVAL_MODES,
)
from helpdeskai.rag.workflow import run_evaluation_core, run_rag_core


def _load_env_file(path: Path) -> None:
    """Load .env entries into process environment without overriding existing vars."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> Namespace:
    """Parse CLI arguments for local RAG runs and evaluation."""

    parser = ArgumentParser(description="Run the HelpDeskAI phase-4 RAG workflow")
    parser.add_argument("action", choices=["answer", "evaluate"], default="answer", nargs="?")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection-name", type=str, default="helpdeskai-techqa")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rewrite-model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--generator-model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--retrieval-mode", choices=sorted(VALID_RETRIEVAL_MODES), default="hybrid")
    parser.add_argument("--prompt-variant", choices=sorted(VALID_PROMPT_VARIANTS), default="grounded")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--compression-top-k", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--max-faithfulness-drop", type=float, default=0.05)
    parser.add_argument(
        "--mock-llm",
        action=BooleanOptionalAction,
        default=True,
        help="Mock LLM calls for local validation (disable with --no-mock-llm)",
    )
    return parser.parse_args()


def _build_config(args: Namespace) -> RagConfig:
    return RagConfig(
        chunks_path=args.chunks_path,
        golden_path=args.golden_path,
        evaluation_path=args.evaluation_path,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        rewrite_model=args.rewrite_model,
        reranker_model=args.reranker_model,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        mock_llm=args.mock_llm,
        retrieval_mode=args.retrieval_mode,
        prompt_variant=args.prompt_variant,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        compression_top_k=args.compression_top_k,
        evaluation_sample_size=args.sample_size,
        max_faithfulness_drop=args.max_faithfulness_drop,
    )


def main() -> None:
    _load_env_file(Path(".env"))
    args = parse_args()
    init_logging(log_file=LOG_FILE)
    config = _build_config(args)
    report = run_evaluation_core(
        config,
        sample_size=args.sample_size,
        save_path=args.evaluation_path,
        baseline_path=args.baseline_report,
    )
    summary = report["summary"]
    logging.info("Best prompt variant: %s", summary["best_prompt_variant"])
    logging.info("Best faithfulness: %.4f", summary["best_faithfulness"])
    logging.info("Evaluation report written to %s", args.evaluation_path)


if __name__ == "__main__":
    main()