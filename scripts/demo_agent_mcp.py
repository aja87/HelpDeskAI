from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from helpdeskai.agents.support_agent import IntentDecision, SupportAgent
from helpdeskai.mcp_servers.client import McpServerScripts, StdioMcpClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DemoClassifier:
    """Deterministic demo classifier so this script demonstrates MCP without LLM cost."""

    def __init__(self, *, sensitive: bool = False) -> None:
        self.sensitive = sensitive

    def classify(self, question: str) -> IntentDecision:
        if self.sensitive:
            return IntentDecision("crm_question", "sensitive_action", 0.93, sensitive=True)
        return IntentDecision("crm_question", "crm_support", 0.91)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    console = Console()
    crm = StdioMcpClient(
        scripts=McpServerScripts(
            crm=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "crm.py",
            knowledge=PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "knowledge.py",
        )
    )

    account_agent = SupportAgent.create(intent_classifier=DemoClassifier(), crm_client=crm)
    account = account_agent.ask("Quel est le statut du compte cust_acme ?", thread_id="mcp-account")
    console.print("[bold]Account lookup[/bold]")
    console.print(account["answer"])

    checkpointer = MemorySaver()
    sensitive_agent = SupportAgent.create(
        intent_classifier=DemoClassifier(sensitive=True),
        checkpointer=checkpointer,
        crm_client=crm,
    )
    thread_id = "mcp-ticket"
    initial = sensitive_agent.ask(
        "Cree un ticket urgent pour cust_acme: acces admin bloque depuis ce matin",
        thread_id=thread_id,
    )
    console.print("[bold]Pending action[/bold]")
    console.print(initial["pending_action"])

    approved = sensitive_agent.approve(thread_id=thread_id)
    console.print("[bold]Approved ticket[/bold]")
    console.print(approved["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
