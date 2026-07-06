from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRM_SERVER = PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "crm.py"
KNOWLEDGE_SERVER = PROJECT_ROOT / "helpdeskai" / "mcp_servers" / "knowledge.py"


async def list_tools() -> int:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "crm": {
                "command": sys.executable,
                "args": [str(CRM_SERVER)],
                "transport": "stdio",
            },
            "knowledge": {
                "command": sys.executable,
                "args": [str(KNOWLEDGE_SERVER)],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    console = Console()
    console.print(f"{len(tools)} MCP tools disponibles")
    for tool in sorted(tools, key=lambda item: item.name):
        console.print(f"- {tool.name}: {tool.description or ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List HelpDeskAI MCP tools.")
    parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    return asyncio.run(list_tools())


if __name__ == "__main__":
    raise SystemExit(main())

