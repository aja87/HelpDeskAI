"""MCP client adapter used by the support agent."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helpdeskai.mcp_servers.security import DEFAULT_TOKEN


class McpClientError(RuntimeError):
    """Raised when an MCP tool cannot be called."""


@dataclass(frozen=True)
class McpServerScripts:
    """Stdio MCP server script paths."""

    crm: Path = Path("helpdeskai/mcp_servers/crm.py")
    knowledge: Path = Path("helpdeskai/mcp_servers/knowledge.py")


class StdioMcpClient:
    """Small synchronous facade over M08's MultiServerMCPClient."""

    def __init__(
        self,
        *,
        scripts: McpServerScripts | None = None,
        token: str = DEFAULT_TOKEN,
        actor_id: str = "agent_default",
    ) -> None:
        self.scripts = scripts or McpServerScripts()
        self.token = token
        self.actor_id = actor_id

    def _server_config(self) -> dict[str, dict[str, Any]]:
        return {
            "crm": {
                "command": sys.executable,
                "args": [str(self.scripts.crm)],
                "transport": "stdio",
            },
            "knowledge": {
                "command": sys.executable,
                "args": [str(self.scripts.knowledge)],
                "transport": "stdio",
            },
        }

    async def _call_async(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(self._server_config())
        tools = await client.get_tools()
        by_name = {tool.name: tool for tool in tools}
        if tool_name not in by_name:
            raise McpClientError(f"MCP tool not found: {tool_name}")
        payload = args | {"token": self.token, "actor_id": self.actor_id}
        result = await by_name[tool_name].ainvoke(payload)
        return self._parse_tool_result(result)

    def _parse_tool_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and first.get("type") == "text":
                try:
                    parsed = json.loads(str(first.get("text", "")))
                except json.JSONDecodeError:
                    return {"result": first.get("text")}
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            return {"result": result}
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                return {"result": result}
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        return {"result": result}

    def call_tool(self, tool_name: str, **args: Any) -> dict[str, Any]:
        try:
            return asyncio.run(self._call_async(tool_name, args))
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" not in str(exc):
                raise McpClientError(str(exc)) from exc
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._call_async(tool_name, args))
            finally:
                loop.close()
        except Exception as exc:
            raise McpClientError(str(exc)) from exc

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        return self.call_tool("get_customer", customer_id=customer_id)

    def get_subscription_status(self, customer_id: str) -> dict[str, Any]:
        return self.call_tool("get_subscription_status", customer_id=customer_id)

    def list_recent_tickets(self, customer_id: str, limit: int = 5) -> dict[str, Any]:
        return self.call_tool("list_recent_tickets", customer_id=customer_id, limit=limit)

    def search_knowledge(
        self,
        query: str,
        *,
        top_k: int = 5,
        product: str | None = None,
        version: str | None = None,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        return self.call_tool(
            "search_knowledge",
            query=query,
            top_k=top_k,
            product=product,
            version=version,
            tenant=tenant,
        )

    def create_ticket(
        self,
        *,
        customer_id: str,
        subject: str,
        body: str,
        priority: str = "medium",
    ) -> dict[str, Any]:
        return self.call_tool(
            "create_ticket",
            customer_id=customer_id,
            subject=subject,
            body=body,
            priority=priority,
        )
