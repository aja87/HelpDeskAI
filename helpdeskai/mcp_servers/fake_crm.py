"""Offline mock CRM data and business operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from helpdeskai.mcp_servers.security import audited_tool


class Priority(StrEnum):
    """Allowed support ticket priorities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CustomerIdInput(BaseModel):
    customer_id: str = Field(pattern=r"^cust_[a-z0-9_]{2,40}$")


class RecentTicketsInput(CustomerIdInput):
    limit: int = Field(default=5, ge=1, le=20)


class CreateTicketInput(CustomerIdInput):
    subject: str = Field(min_length=5, max_length=120)
    body: str = Field(min_length=10, max_length=2000)
    priority: Priority = Priority.MEDIUM


CUSTOMERS: dict[str, dict[str, Any]] = {
    "cust_acme": {
        "customer_id": "cust_acme",
        "name": "Acme Europe",
        "tenant": "acme-prod",
        "contacts": ["ops@acme.example"],
    },
    "cust_globex": {
        "customer_id": "cust_globex",
        "name": "Globex France",
        "tenant": "globex-prod",
        "contacts": ["it@globex.example"],
    },
}

SUBSCRIPTIONS: dict[str, dict[str, Any]] = {
    "cust_acme": {
        "customer_id": "cust_acme",
        "status": "active",
        "plan": "enterprise",
        "product": "NovaCloud",
        "seats_used": 84,
        "seats_total": 120,
        "renewal_date": "2026-11-30",
    },
    "cust_globex": {
        "customer_id": "cust_globex",
        "status": "past_due",
        "plan": "business",
        "product": "NovaCloud",
        "seats_used": 41,
        "seats_total": 50,
        "renewal_date": "2026-08-15",
    },
}

TICKETS: list[dict[str, Any]] = [
    {
        "ticket_id": "TCK-1001",
        "customer_id": "cust_acme",
        "subject": "SAML metadata renewal",
        "priority": "medium",
        "status": "open",
        "created_at": "2026-06-18T09:20:00Z",
    },
    {
        "ticket_id": "TCK-1002",
        "customer_id": "cust_acme",
        "subject": "Admin console access",
        "priority": "high",
        "status": "waiting_customer",
        "created_at": "2026-06-27T14:08:00Z",
    },
    {
        "ticket_id": "TCK-2001",
        "customer_id": "cust_globex",
        "subject": "Billing status review",
        "priority": "medium",
        "status": "open",
        "created_at": "2026-06-29T11:45:00Z",
    },
]


def _not_found(customer_id: str) -> dict[str, Any]:
    return {"error": "customer_not_found", "customer_id": customer_id}


@audited_tool("get_customer")
def get_customer_business(*, actor_id: str, customer_id: str) -> dict[str, Any]:
    args = CustomerIdInput(customer_id=customer_id)
    return CUSTOMERS.get(args.customer_id, _not_found(args.customer_id))


@audited_tool("get_subscription_status")
def get_subscription_status_business(*, actor_id: str, customer_id: str) -> dict[str, Any]:
    args = CustomerIdInput(customer_id=customer_id)
    if args.customer_id not in CUSTOMERS:
        return _not_found(args.customer_id)
    return SUBSCRIPTIONS[args.customer_id]


@audited_tool("list_recent_tickets")
def list_recent_tickets_business(
    *,
    actor_id: str,
    customer_id: str,
    limit: int = 5,
) -> dict[str, Any]:
    args = RecentTicketsInput(customer_id=customer_id, limit=limit)
    if args.customer_id not in CUSTOMERS:
        return _not_found(args.customer_id)
    tickets = [ticket for ticket in TICKETS if ticket["customer_id"] == args.customer_id]
    tickets = sorted(tickets, key=lambda item: item["created_at"], reverse=True)[: args.limit]
    return {"customer_id": args.customer_id, "count": len(tickets), "tickets": tickets}


@audited_tool("create_ticket")
def create_ticket_business(
    *,
    actor_id: str,
    customer_id: str,
    subject: str,
    body: str,
    priority: str = "medium",
) -> dict[str, Any]:
    args = CreateTicketInput(
        customer_id=customer_id,
        subject=subject,
        body=body,
        priority=priority,
    )
    if args.customer_id not in CUSTOMERS:
        return _not_found(args.customer_id)
    ticket_id = f"TCK-{1000 + len(TICKETS) + 1}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": args.customer_id,
        "subject": args.subject,
        "body": args.body,
        "priority": args.priority.value,
        "status": "open",
        "created_by": actor_id,
    }
    TICKETS.insert(0, ticket)
    return {"success": True, "ticket": ticket}

