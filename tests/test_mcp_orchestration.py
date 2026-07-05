from __future__ import annotations

import json
from pathlib import Path

from helpdeskai.agents.mcp_workflow import run_mcp_agent_core
from helpdeskai.mcp_servers.crm import CRMService
from helpdeskai.mcp_servers.knoweldge import KnowledgeService


TOKEN = "unit-test-token"


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "C1",
            "doc_id": "D1",
            "text": "Invoices are available from the billing dashboard under Finance.",
            "product": "Core",
            "version": "1.0",
            "category": "billing",
            "source": "techqa",
        },
        {
            "chunk_id": "C2",
            "doc_id": "D2",
            "text": "To reset credentials, open admin console then security panel.",
            "product": "Core",
            "version": "1.0",
            "category": "security",
            "source": "techqa",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_billing_query_checks_subscription_then_answers(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    crm = CRMService(expected_token=TOKEN)
    knowledge = KnowledgeService(chunks_path=chunks_path, expected_token=TOKEN)

    payload = run_mcp_agent_core(
        query="Where can I find my billing invoice?",
        customer_id="CUST-100",
        token=TOKEN,
        crm_service=crm,
        knowledge_service=knowledge,
    )

    assert payload["path_taken"] == ["classify", "check_subscription", "knowledge"]
    assert payload["subscription"]["subscription_status"] == "active"
    assert payload["answer"]


def test_billing_query_stops_when_subscription_inactive(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    crm = CRMService(expected_token=TOKEN)
    knowledge = KnowledgeService(chunks_path=chunks_path, expected_token=TOKEN)

    payload = run_mcp_agent_core(
        query="I need billing support for my renewal",
        customer_id="CUST-200",
        token=TOKEN,
        crm_service=crm,
        knowledge_service=knowledge,
    )

    assert payload["path_taken"] == ["classify", "check_subscription", "inactive_subscription"]
    assert payload["subscription"]["subscription_status"] == "past_due"
    assert "must be active" in payload["answer"]


def test_non_billing_query_skips_crm_check(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    crm = CRMService(expected_token=TOKEN)
    knowledge = KnowledgeService(chunks_path=chunks_path, expected_token=TOKEN)

    payload = run_mcp_agent_core(
        query="How can I reset credentials?",
        customer_id="CUST-100",
        token=TOKEN,
        crm_service=crm,
        knowledge_service=knowledge,
    )

    assert payload["path_taken"] == ["classify", "knowledge"]
    assert payload["subscription"] == {}
    assert payload["answer"]
