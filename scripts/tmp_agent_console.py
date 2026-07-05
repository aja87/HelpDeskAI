"""Temporary interactive console to test the HelpDeskAI support agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpdeskai.agents import AgentConfig, SupportAgent, open_sqlite_checkpointer  # noqa: E402
from helpdeskai.agents.support_agent import IntentClassificationError  # noqa: E402
from helpdeskai.mcp_servers.client import McpServerScripts, StdioMcpClient  # noqa: E402
from helpdeskai.rag.llm import MissingAnthropicKeyError  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default="console-agent")
    parser.add_argument("--checkpoint-db", type=Path, default=Path("data/tmp_agent_console.sqlite"))
    parser.add_argument("--with-mcp", action="store_true", help="Enable CRM MCP calls.")
    parser.add_argument("--mcp-token", default="helpdeskai-dev-token")
    return parser.parse_args(argv)


def print_state(state: dict) -> None:
    console.print(Panel(state.get("answer", ""), title="Answer", border_style="orange3"))
    if state.get("pending_action"):
        console.print(
            Panel(str(state["pending_action"]), title="Pending action", border_style="yellow")
        )
        console.print("[dim]Type /approve or /reject to resume this action.[/dim]")
    if state.get("sources"):
        console.print(f"[dim]Sources: {', '.join(state['sources'])}[/dim]")
    console.print(f"[dim]Path: {' -> '.join(state.get('path_taken', []))}[/dim]")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    crm_client = None
    if args.with_mcp:
        crm_client = StdioMcpClient(
            scripts=McpServerScripts(
                crm=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "crm.py",
                knowledge=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "knowledge.py",
            ),
            token=args.mcp_token,
        )

    console.print("[bold]HelpDeskAI console[/bold]")
    console.print("[dim]Commands: /approve, /reject, /quit[/dim]\n")

    try:
        with open_sqlite_checkpointer(args.checkpoint_db) as checkpointer:
            agent = SupportAgent.create(
                config=AgentConfig(),
                checkpointer=checkpointer,
                crm_client=crm_client,
            )
            while True:
                question = console.input("[cyan]> [/cyan]").strip()
                if not question:
                    continue
                if question in {"/quit", "/exit"}:
                    return 0
                try:
                    if question in {"/approve", "approve"}:
                        state = agent.approve(thread_id=args.thread_id)
                    elif question in {"/reject", "reject"}:
                        state = agent.reject(thread_id=args.thread_id)
                    else:
                        state = agent.ask(question, thread_id=args.thread_id)
                except IntentClassificationError as exc:
                    console.print(f"[red]error: {exc}[/red]")
                    continue
                print_state(state)
    except MissingAnthropicKeyError as exc:
        console.print(f"[red]error: {exc}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
