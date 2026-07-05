from __future__ import annotations

import pytest

from helpdeskai.mcp_servers.crm import CRMService, MCPAuthError, MCPRateLimitError, ToolInputError


TOKEN = "unit-test-token"


def test_get_customer_returns_profile() -> None:
    service = CRMService(expected_token=TOKEN)

    payload = service.get_customer(customer_id="CUST-100", token=TOKEN)

    assert payload["customer_id"] == "CUST-100"
    assert payload["subscription_status"] == "active"


def test_authentication_is_enforced() -> None:
    service = CRMService(expected_token=TOKEN)

    with pytest.raises(MCPAuthError):
        service.get_subscription_status(customer_id="CUST-100", token="wrong-token")


def test_create_ticket_validates_inputs_strictly() -> None:
    service = CRMService(expected_token=TOKEN)

    with pytest.raises(ToolInputError):
        service.create_ticket(
            customer_id="CUST-100",
            subject="bad",
            body="short",
            priority="urgent",
            token=TOKEN,
        )


def test_rate_limiting_blocks_after_threshold() -> None:
    service = CRMService(expected_token=TOKEN, rate_limit_calls=2, rate_limit_window_s=60)

    service.get_customer(customer_id="CUST-100", token=TOKEN)
    service.get_customer(customer_id="CUST-100", token=TOKEN)
    with pytest.raises(MCPRateLimitError):
        service.get_customer(customer_id="CUST-100", token=TOKEN)
