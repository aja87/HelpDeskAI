from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class MCPAuthError(PermissionError):
	"""Raised when a tool call is not authenticated."""


class MCPRateLimitError(RuntimeError):
	"""Raised when a token exceeds the local rate limit."""


class ToolInputError(ValueError):
	"""Raised when tool input validation fails."""


class _StrictModel(BaseModel):
	model_config = ConfigDict(extra="forbid", strict=True)


class GetCustomerInput(_StrictModel):
	customer_id: str = Field(min_length=2, max_length=64)
	token: str = Field(min_length=8, max_length=256)


class GetSubscriptionStatusInput(_StrictModel):
	customer_id: str = Field(min_length=2, max_length=64)
	token: str = Field(min_length=8, max_length=256)


class ListRecentTicketsInput(_StrictModel):
	customer_id: str = Field(min_length=2, max_length=64)
	token: str = Field(min_length=8, max_length=256)
	limit: int = Field(default=5, ge=1, le=20)


class CreateTicketInput(_StrictModel):
	customer_id: str = Field(min_length=2, max_length=64)
	subject: str = Field(min_length=4, max_length=200)
	body: str = Field(min_length=8, max_length=4000)
	priority: Literal["low", "medium", "high"] = "medium"
	token: str = Field(min_length=8, max_length=256)


@dataclass(slots=True)
class RateLimiter:
	max_calls: int
	period_seconds: int
	_windows: dict[str, deque[float]] = field(init=False, repr=False)

	def __post_init__(self) -> None:
		self._windows = defaultdict(deque)

	def consume(self, key: str) -> None:
		now = time.monotonic()
		window = self._windows[key]
		threshold = now - float(self.period_seconds)
		while window and window[0] < threshold:
			window.popleft()
		if len(window) >= self.max_calls:
			raise MCPRateLimitError(
				f"Rate limit exceeded for key={key}. max_calls={self.max_calls}, period_seconds={self.period_seconds}"
			)
		window.append(now)


def _sample_customers() -> dict[str, dict[str, Any]]:
	return {
		"CUST-100": {
			"customer_id": "CUST-100",
			"name": "Nova Industries",
			"email": "it-admin@nova-industries.example",
			"plan": "enterprise",
			"subscription_status": "active",
			"renewal_date": "2027-01-15",
		},
		"CUST-200": {
			"customer_id": "CUST-200",
			"name": "Blue Harbor Retail",
			"email": "ops@blue-harbor.example",
			"plan": "pro",
			"subscription_status": "past_due",
			"renewal_date": "2026-03-01",
		},
	}


class CRMService:
	"""In-memory CRM service with strict validation and security guards."""

	TOOL_DOCS = {
		"get_customer": {
			"summary": "Fetch a customer profile from CRM by id.",
			"inputs": {"customer_id": "str", "token": "str"},
			"returns": "dict customer profile or error payload",
		},
		"get_subscription_status": {
			"summary": "Fetch billing/subscription status by customer id.",
			"inputs": {"customer_id": "str", "token": "str"},
			"returns": "dict with subscription status fields",
		},
		"list_recent_tickets": {
			"summary": "List latest support tickets linked to a customer.",
			"inputs": {"customer_id": "str", "token": "str", "limit": "int=5"},
			"returns": "list[dict] of ticket rows",
		},
		"create_ticket": {
			"summary": "Create a support ticket in CRM for a customer.",
			"inputs": {
				"customer_id": "str",
				"subject": "str",
				"body": "str",
				"priority": "low|medium|high",
				"token": "str",
			},
			"returns": "dict ticket payload",
		},
	}

	def __init__(
		self,
		*,
		expected_token: str | None = None,
		rate_limit_calls: int = 60,
		rate_limit_window_s: int = 60,
	) -> None:
		self.expected_token = expected_token or os.getenv("HELPDESKAI_MCP_TOKEN", "dev-token-unsafe")
		self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, period_seconds=rate_limit_window_s)
		self.customers = _sample_customers()
		self.tickets: list[dict[str, Any]] = [
			{
				"ticket_id": "TKT-001",
				"customer_id": "CUST-100",
				"subject": "Invoice clarification",
				"body": "Need invoice line-item details for March.",
				"priority": "medium",
				"status": "open",
				"created_at": "2026-06-29T09:12:00+00:00",
			},
			{
				"ticket_id": "TKT-002",
				"customer_id": "CUST-200",
				"subject": "Subscription failed renewal",
				"body": "Payment was rejected on renewal date.",
				"priority": "high",
				"status": "open",
				"created_at": "2026-07-01T16:33:00+00:00",
			},
		]

	@staticmethod
	def describe_tools() -> dict[str, Any]:
		return CRMService.TOOL_DOCS

	def _authorize(self, token: str) -> None:
		if token != self.expected_token:
			raise MCPAuthError("Invalid MCP token")
		self.rate_limiter.consume(token)

	@staticmethod
	def _validate(model: type[_StrictModel], payload: dict[str, Any]) -> _StrictModel:
		try:
			return model.model_validate(payload)
		except ValidationError as exc:
			raise ToolInputError(str(exc)) from exc

	def get_customer(self, *, customer_id: str, token: str) -> dict[str, Any]:
		req = self._validate(GetCustomerInput, {"customer_id": customer_id, "token": token})
		self._authorize(req.token)
		customer = self.customers.get(req.customer_id)
		if customer is None:
			return {"error": "customer_not_found", "customer_id": req.customer_id}
		return dict(customer)

	def get_subscription_status(self, *, customer_id: str, token: str) -> dict[str, Any]:
		req = self._validate(GetSubscriptionStatusInput, {"customer_id": customer_id, "token": token})
		self._authorize(req.token)
		customer = self.customers.get(req.customer_id)
		if customer is None:
			return {"error": "customer_not_found", "customer_id": req.customer_id}
		return {
			"customer_id": req.customer_id,
			"plan": customer["plan"],
			"subscription_status": customer["subscription_status"],
			"renewal_date": customer["renewal_date"],
		}

	def list_recent_tickets(self, *, customer_id: str, token: str, limit: int = 5) -> list[dict[str, Any]]:
		req = self._validate(
			ListRecentTicketsInput,
			{"customer_id": customer_id, "token": token, "limit": limit},
		)
		self._authorize(req.token)
		rows = [row for row in self.tickets if row["customer_id"] == req.customer_id]
		rows.sort(key=lambda row: row["created_at"], reverse=True)
		return [dict(row) for row in rows[: req.limit]]

	def create_ticket(
		self,
		*,
		customer_id: str,
		subject: str,
		body: str,
		priority: str,
		token: str,
	) -> dict[str, Any]:
		req = self._validate(
			CreateTicketInput,
			{
				"customer_id": customer_id,
				"subject": subject,
				"body": body,
				"priority": priority,
				"token": token,
			},
		)
		self._authorize(req.token)
		if req.customer_id not in self.customers:
			return {"error": "customer_not_found", "customer_id": req.customer_id}
		ticket_id = f"TKT-{len(self.tickets) + 1:03d}"
		created = {
			"ticket_id": ticket_id,
			"customer_id": req.customer_id,
			"subject": req.subject,
			"body": req.body,
			"priority": req.priority,
			"status": "open",
			"created_at": datetime.now(UTC).isoformat(),
		}
		self.tickets.append(created)
		return dict(created)


def _build_mcp_app(service: CRMService):
	try:
		from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]
	except Exception as exc:  # pragma: no cover
		logging.exception("Failed to import FastMCP for CRM server")
		raise RuntimeError("mcp package is required to run the CRM MCP server") from exc

	mcp = FastMCP("crm-server")

	@mcp.tool()
	def get_customer(customer_id: str, token: str) -> dict[str, Any]:
		"""Get customer profile data by customer ID."""

		return service.get_customer(customer_id=customer_id, token=token)

	@mcp.tool()
	def get_subscription_status(customer_id: str, token: str) -> dict[str, Any]:
		"""Get current subscription status for a customer."""

		return service.get_subscription_status(customer_id=customer_id, token=token)

	@mcp.tool()
	def list_recent_tickets(customer_id: str, token: str, limit: int = 5) -> list[dict[str, Any]]:
		"""List recent tickets linked to a customer."""

		return service.list_recent_tickets(customer_id=customer_id, token=token, limit=limit)

	@mcp.tool()
	def create_ticket(customer_id: str, subject: str, body: str, priority: str, token: str) -> dict[str, Any]:
		"""Create a new customer ticket in the CRM."""

		return service.create_ticket(
			customer_id=customer_id,
			subject=subject,
			body=body,
			priority=priority,
			token=token,
		)

	return mcp


def run_server() -> None:
	logging.info("Initializing CRM service")
	service = CRMService()
	logging.info("CRM service ready: customers=%d tickets=%d", len(service.customers), len(service.tickets))
	app = _build_mcp_app(service)
	logging.info("CRM MCP server bootstrapped; entering FastMCP run loop")
	app.run()


if __name__ == "__main__":
	run_server()
