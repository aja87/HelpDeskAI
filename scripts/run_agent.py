"""Run the Phase 5 HelpDeskAI LangGraph support agent."""

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
from helpdeskai.rag.llm import MissingAnthropicKeyError  # noqa: E402

console = Console()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    parser.add_argument("--thread-id", default="demo-agent")
    parser.add_argument("--checkpoint-db", type=Path, default=Path("data/agent_checkpoints.sqlite"))
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve a pending sensitive action.",
    )
    parser.add_argument("--reject", action="store_true", help="Reject a pending sensitive action.")
    parser.add_argument("--export-mermaid", type=Path, help="Write the graph Mermaid diagram.")
    return parser.parse_args(argv)


def _print_state(state: dict) -> None:
    console.print(Panel(state.get("answer", ""), title="Agent answer", border_style="orange3"))
    if state.get("pending_action"):
        console.print(
            Panel(str(state["pending_action"]), title="Pending action", border_style="yellow")
        )
    if state.get("sources"):
        console.print(f"[dim]Sources: {', '.join(state['sources'])}[/dim]")
    console.print(f"[dim]Path: {' -> '.join(state.get('path_taken', []))}[/dim]")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        with open_sqlite_checkpointer(args.checkpoint_db) as checkpointer:
            agent = SupportAgent.create(config=AgentConfig(), checkpointer=checkpointer)
            if args.export_mermaid:
                args.export_mermaid.parent.mkdir(parents=True, exist_ok=True)
                args.export_mermaid.write_text(agent.draw_mermaid(), encoding="utf-8")
                console.print(f"[green]Mermaid exported to {args.export_mermaid}[/green]")
                if not args.question and not args.approve and not args.reject:
                    return 0

            if args.approve:
                state = agent.approve(thread_id=args.thread_id)
            elif args.reject:
                state = agent.reject(thread_id=args.thread_id)
            else:
                question = args.question or "How do I configure SAML login in NovaCloud?"
                state = agent.ask(question, thread_id=args.thread_id)
            _print_state(state)
    except MissingAnthropicKeyError as exc:
        console.print(f"[red]error: {exc}[/red]")
        return 1
    except IntentClassificationError as exc:
        console.print(f"[red]error: {exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
