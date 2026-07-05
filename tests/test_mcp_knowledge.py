from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpdeskai.mcp_servers.knoweldge import (
    KnowledgeService,
    MCPAuthError,
    MCPRateLimitError,
    ToolInputError,
)


TOKEN = "unit-test-token"


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "C1",
            "doc_id": "D1",
            "text": "Subscription renewal and invoice details are available in billing settings.",
            "product": "Core",
            "version": "1.0",
            "category": "billing",
            "source": "techqa",
        },
        {
            "chunk_id": "C2",
            "doc_id": "D2",
            "text": "Reset your password from the admin console under security options.",
            "product": "Core",
            "version": "1.0",
            "category": "security",
            "source": "techqa",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_search_knowledge_returns_ranked_hits(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    service = KnowledgeService(chunks_path=chunks_path, expected_token=TOKEN)

    hits = service.search_knowledge(query="billing invoice", token=TOKEN, top_k=1)

    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "C1"


def test_search_knowledge_enforces_auth(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    service = KnowledgeService(chunks_path=chunks_path, expected_token=TOKEN)

    with pytest.raises(MCPAuthError):
        service.search_knowledge(query="billing", token="bad-token", top_k=1)


def test_answer_question_validates_inputs(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    service = KnowledgeService(chunks_path=chunks_path, expected_token=TOKEN)

    with pytest.raises(ToolInputError):
        service.answer_question(query="ok", token=TOKEN, top_k=3)


def test_rate_limiting_blocks_after_threshold(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    service = KnowledgeService(
        chunks_path=chunks_path,
        expected_token=TOKEN,
        rate_limit_calls=1,
        rate_limit_window_s=60,
    )

    service.search_knowledge(query="billing", token=TOKEN, top_k=1)
    with pytest.raises(MCPRateLimitError):
        service.search_knowledge(query="billing", token=TOKEN, top_k=1)
