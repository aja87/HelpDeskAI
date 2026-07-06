"""FastMCP CRM server for HelpDeskAI."""

from __future__ import annotations

import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from helpdeskai.mcp_servers.fake_crm import (
    create_ticket_business,
    get_customer_business,
    get_subscription_status_business,
    list_recent_tickets_business,
)

mcp = FastMCP(
    "helpdeskai-crm",
    host=os.environ.get("HELPDESKAI_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("HELPDESKAI_MCP_PORT", "8000")),
)


@mcp.tool()
def get_customer(customer_id: str, token: str, actor_id: str = "agent_default") -> dict[str, Any]:
    """Return mock CRM customer identity."""
    return get_customer_business(actor_id=actor_id, token=token, customer_id=customer_id)


@mcp.tool()
def get_subscription_status(
    customer_id: str,
    token: str,
    actor_id: str = "agent_default",
) -> dict[str, Any]:
    """Return mock subscription status for one customer."""
    return get_subscription_status_business(actor_id=actor_id, token=token, customer_id=customer_id)


@mcp.tool()
def list_recent_tickets(
    customer_id: str,
    token: str,
    limit: int = 5,
    actor_id: str = "agent_default",
) -> dict[str, Any]:
    """List recent mock support tickets for one customer."""
    return list_recent_tickets_business(
        actor_id=actor_id,
        token=token,
        customer_id=customer_id,
        limit=limit,
    )


@mcp.tool()
def create_ticket(
    customer_id: str,
    subject: str,
    body: str,
    token: str,
    priority: str = "medium",
    actor_id: str = "agent_default",
) -> dict[str, Any]:
    """Create a mock CRM ticket after human approval."""
    return create_ticket_business(
        actor_id=actor_id,
        token=token,
        customer_id=customer_id,
        subject=subject,
        body=body,
        priority=priority,
    )


if __name__ == "__main__":
    if "--transport" in sys.argv:
        transport = sys.argv[sys.argv.index("--transport") + 1]
        mcp.run(transport=transport)
    else:
        mcp.run()
