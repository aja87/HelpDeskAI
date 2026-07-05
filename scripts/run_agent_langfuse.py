"""Run the HelpDeskAI agent with Langfuse tracing callbacks."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.agents import AgentConfig, SupportAgent, open_sqlite_checkpointer  # noqa: E402
from helpdeskai.mcp_servers.client import McpServerScripts, StdioMcpClient  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", action="append")
    parser.add_argument("--thread-id", default="langfuse-demo")
    parser.add_argument("--user-id", default="demo")
    parser.add_argument("--checkpoint-db", type=Path, default=Path("data/agent_checkpoints.sqlite"))
    parser.add_argument("--with-mcp", action="store_true")
    parser.add_argument("--mcp-token", default="helpdeskai-dev-token")
    parser.add_argument(
        "--crm-server",
        type=Path,
        default=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "crm.py",
    )
    parser.add_argument(
        "--knowledge-server",
        type=Path,
        default=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "knowledge.py",
    )
    return parser.parse_args(argv)


def _questions(args: argparse.Namespace) -> list[str]:
    return args.question or [
        "How do I configure SAML login in NovaCloud?",
        "Quel est le statut de cust_acme ?",
        "Escalade le compte cust_acme pour acces admin bloque",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from dotenv import load_dotenv
    try:
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError:
        console.print(
            "[red]error: missing optional dependency `langfuse`.[/red]\n"
            "Install the Phase 7 dependencies first, for example:\n"
            "  python -m ensurepip --upgrade\n"
            "  python -m pip install \"langfuse>=3\"\n"
            "or, if uv is installed:\n"
            "  uv sync --dev"
        )
        return 1

    load_dotenv(PROJECT_ROOT / ".env")
    handler = CallbackHandler()
    session_id = f"session_{uuid.uuid4()}"
    args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)

    with open_sqlite_checkpointer(args.checkpoint_db) as checkpointer:
        crm_client = None
        if args.with_mcp:
            crm_client = StdioMcpClient(
                scripts=McpServerScripts(crm=args.crm_server, knowledge=args.knowledge_server),
                token=args.mcp_token,
            )
        agent = SupportAgent.create(
            config=AgentConfig(),
            checkpointer=checkpointer,
            crm_client=crm_client,
        )
        for index, question in enumerate(_questions(args), start=1):
            config = {
                "callbacks": [handler],
                "configurable": {"thread_id": f"{args.thread_id}-{index}"},
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": args.user_id,
                },
            }
            state = {
                "question": question,
                "approval": None,
                "pending_action": None,
                "iterations": 0,
                "tokens_used": 0,
                "path_taken": [],
            }
            output = agent.graph.invoke(state, config=config)
            console.print(Panel(output.get("answer", ""), title=question, border_style="cyan"))
            if output.get("pending_action"):
                agent.graph.update_state(config, {"approval": "approved"})
                approved = agent.graph.invoke(None, config=config)
                console.print(
                    Panel(
                        approved.get("answer", ""),
                        title="Approved sensitive action",
                        border_style="yellow",
                    )
                )
    flush = getattr(handler, "flush", None)
    if callable(flush):
        flush()
    console.print("[green]Open Langfuse at http://localhost:3000 -> Tracing -> Traces[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
