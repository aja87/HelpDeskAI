
from __future__ import annotations

import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

from helpdeskai.agents.config import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_GRAPH_PATH,
    LOG_FILE,
    AgentsConfig,
)
from helpdeskai.agents.workflow import export_graph_core, run_agents_core
from helpdeskai.common.logging import init_logging


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
    """Parse CLI arguments for local agents runs."""

    parser = ArgumentParser(description="Run the HelpDeskAI phase-5 agents workflow")
    parser.add_argument("action", choices=["ask", "export-graph"], default="ask", nargs="?")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--session-id", type=str, default="local-session")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--checkpoint-backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--checkpoint-dsn", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument(
        "--mock-llm",
        action=BooleanOptionalAction,
        default=True,
        help="Mock LLM calls for local validation (disable with --no-mock-llm)",
    )
    parser.add_argument("--classifier-model", type=str, default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--generator-model", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection-name", type=str, default="helpdeskai-techqa")
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH)
    return parser.parse_args()


def _build_config(args: Namespace) -> AgentsConfig:
    return AgentsConfig(
        chunks_path=args.chunks_path,
        checkpoint_backend=args.checkpoint_backend,
        checkpoint_dsn=args.checkpoint_dsn,
        session_id=args.session_id,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        classifier_model=args.classifier_model,
        generator_model=args.generator_model,
        mock_llm=args.mock_llm,
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        graph_path=args.graph_path,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def main() -> None:
    """CLI entrypoint for local agents orchestration runs."""

    _load_env_file(Path(".env"))
    args = parse_args()
    if args.action == "ask" and not args.query:
        raise ValueError("--query is required when action=ask")
    init_logging(log_file=LOG_FILE)

    config = _build_config(args)

    if args.action == "export-graph":
        payload = export_graph_core(config, output_path=args.graph_path)
        logging.info("Workflow graph exported to %s", payload["graph_path"])
        return

    payload = run_agents_core(config, query=args.query or "", session_id=args.session_id)

    if isinstance(payload, dict):
        logging.info("Agent action completed: ask")
        if "session_id" in payload:
            logging.info("Session: %s", payload["session_id"])
        if "status" in payload:
            logging.info("Status: %s", payload["status"])
        if "answer" in payload:
            logging.info("Answer: %s", payload["answer"])
        if "clarification" in payload:
            logging.info("Clarification: %s", payload["clarification"])
        if "graph_path" in payload:
            logging.info("Graph exported to %s", payload["graph_path"])
    elif payload is not None:
        logging.info("Agent workflow output: %s", payload)


if __name__ == "__main__":
    main()