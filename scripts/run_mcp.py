
from __future__ import annotations

import json
import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

from helpdeskai.agents.mcp_workflow import run_mcp_agent_core
from helpdeskai.common.logging import init_logging
from helpdeskai.mcp_servers.crm import CRMService, run_server as run_crm_server
from helpdeskai.mcp_servers.knoweldge import KnowledgeService, run_server as run_knowledge_server

LOG_FILE = "mcp.log"


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
    """Parse CLI arguments for phase-6 MCP servers and orchestration."""

    parser = ArgumentParser(description="Run the HelpDeskAI phase-6 MCP workflow")
    parser.add_argument(
        "action",
        nargs="?",
        default="ask",
        choices=["ask", "serve-crm", "serve-knowledge", "describe-tools"],
    )
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--customer-id", type=str, default=None)
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/techqa_chunks.jsonl"))
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--pretty",
        action=BooleanOptionalAction,
        default=True,
        help="Pretty-print JSON outputs.",
    )
    return parser.parse_args()


def _json_dump(payload: dict[str, object], *, pretty: bool) -> str:
    return json.dumps(payload, indent=2 if pretty else None, ensure_ascii=True)


def _resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("HELPDESKAI_MCP_TOKEN")
    if not token:
        raise ValueError("MCP token is required. Set --token or HELPDESKAI_MCP_TOKEN.")
    return token


def main() -> None:
    _load_env_file(Path(".env"))
    args = parse_args()
    init_logging(log_file=LOG_FILE)
    logging.info("MCP entrypoint starting with action=%s", args.action)
    if args.action == "ask":
        logging.info("MCP ask params: customer_id=%s top_k=%s chunks_path=%s", args.customer_id, args.top_k, args.chunks_path)
    try:
        if args.action == "serve-crm":
            logging.info("Starting CRM MCP server")
            run_crm_server()
            return

        if args.action == "serve-knowledge":
            logging.info("Starting Knowledge MCP server")
            run_knowledge_server()
            return

        token = _resolve_token(args.token)

        if args.action == "describe-tools":
            payload = {
                "crm": CRMService.describe_tools(),
                "knowledge": KnowledgeService.describe_tools(),
            }
            print(_json_dump(payload, pretty=args.pretty))
            return

        if not args.query:
            raise ValueError("--query is required when action=ask")

        crm_service = CRMService(expected_token=token)
        knowledge_service = KnowledgeService(chunks_path=args.chunks_path, expected_token=token)
        payload = run_mcp_agent_core(
            query=args.query,
            token=token,
            customer_id=args.customer_id,
            top_k=args.top_k,
            crm_service=crm_service,
            knowledge_service=knowledge_service,
        )
        logging.info("MCP orchestration path: %s", payload.get("path_taken"))
        print(_json_dump(payload, pretty=args.pretty))
    except Exception:
        logging.exception("MCP run failed for action=%s", args.action)
        raise


if __name__ == "__main__":
    main()
    