from __future__ import annotations

from helpdeskai.mcp_servers import fake_crm, knowledge
from helpdeskai.mcp_servers.security import DEFAULT_TOKEN, reset_rate_limits
from helpdeskai.retrieval.models import SearchMode, SearchResult


def setup_function() -> None:
    reset_rate_limits()


def test_crm_get_customer_valid() -> None:
    result = fake_crm.get_customer_business(
        actor_id="test",
        token=DEFAULT_TOKEN,
        customer_id="cust_acme",
    )

    assert result["name"] == "Acme Europe"


def test_crm_get_subscription_status_valid() -> None:
    result = fake_crm.get_subscription_status_business(
        actor_id="test",
        token=DEFAULT_TOKEN,
        customer_id="cust_acme",
    )

    assert result["status"] == "active"
    assert result["seats_total"] == 120


def test_crm_list_recent_tickets() -> None:
    result = fake_crm.list_recent_tickets_business(
        actor_id="test",
        token=DEFAULT_TOKEN,
        customer_id="cust_acme",
        limit=1,
    )

    assert result["count"] == 1
    assert result["tickets"][0]["customer_id"] == "cust_acme"


def test_crm_create_ticket_validates_priority() -> None:
    invalid = fake_crm.create_ticket_business(
        actor_id="test",
        token=DEFAULT_TOKEN,
        customer_id="cust_acme",
        subject="Ticket test",
        body="Creation d'un ticket de test",
        priority="critical",
    )

    assert invalid["error"] == "validation_error"

    valid = fake_crm.create_ticket_business(
        actor_id="test",
        token=DEFAULT_TOKEN,
        customer_id="cust_acme",
        subject="Ticket test",
        body="Creation d'un ticket de test",
        priority="high",
    )

    assert valid["success"] is True
    assert valid["ticket"]["priority"] == "high"


def test_crm_invalid_token_rejected() -> None:
    result = fake_crm.get_customer_business(
        actor_id="test",
        token="bad-token",
        customer_id="cust_acme",
    )

    assert result["error"] == "unauthorized"


def test_crm_rate_limit_returns_structured_error() -> None:
    result = {}
    for _ in range(11):
        result = fake_crm.get_customer_business(
            actor_id="limited",
            token=DEFAULT_TOKEN,
            customer_id="cust_acme",
        )

    assert result["error"] == "rate_limited"
    assert result["retry_after_s"] >= 1


def test_knowledge_validates_query_and_top_k() -> None:
    result = knowledge.search_knowledge_business(actor_id="test", query="no", top_k=5)

    assert result["error"] == "validation_error"

    result = knowledge.search_knowledge_business(actor_id="test", query="NovaCloud SAML", top_k=99)

    assert result["error"] == "validation_error"


def test_knowledge_calls_injected_backend_and_returns_sources() -> None:
    calls = []

    def fake_backend(query, top_k, filters):
        calls.append((query, top_k, filters))
        return [
            SearchResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="NovaCloud SAML setup details",
                score=0.88,
                mode=SearchMode.HYBRID,
                metadata={"product": "NovaCloud"},
                source_scores={"sparse": 0.7},
            )
        ]

    result = knowledge.search_knowledge_business(
        actor_id="test",
        query="NovaCloud SAML",
        top_k=1,
        product="NovaCloud",
        backend=fake_backend,
    )

    assert calls[0][0] == "NovaCloud SAML"
    assert calls[0][1] == 1
    assert calls[0][2].product == "NovaCloud"
    assert result["results"][0]["source_id"] == "chunk-1"
