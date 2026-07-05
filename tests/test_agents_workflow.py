from __future__ import annotations

import json
from pathlib import Path

from helpdeskai.agents.config import AgentsConfig
from helpdeskai.agents.workflow import run_agents_core
from helpdeskai.retrieval.workflow import SearchHit


class FakeRetrievalEngine:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: object | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        del top_k, filters, mode
        self.queries.append(query)
        return self.hits


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, object]] = {}

    def load(self, session_id: str) -> dict[str, object] | None:
        return self.data.get(session_id)

    def save(self, session_id: str, state: dict[str, object]) -> None:
        self.data[session_id] = json.loads(json.dumps(state))


def _write_chunks(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _hits() -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id="C1",
            doc_id="D1",
            score=0.9,
            text="Reset the queue manager password from the admin console.",
            source="techqa",
            product="MQ",
            version="9.3",
            category="troubleshooting",
            date="2024-01-01",
        ),
        SearchHit(
            chunk_id="C2",
            doc_id="D2",
            score=0.5,
            text="Review audit settings after you change any administrator credential.",
            source="techqa",
            product="MQ",
            version="9.3",
            category="operations",
            date="2024-01-01",
        ),
    ]


def _build_config(tmp_path: Path) -> AgentsConfig:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            {
                "chunk_id": "C1",
                "doc_id": "D1",
                "text": "Reset the queue manager password from the admin console.",
                "source": "techqa",
                "product": "MQ",
                "version": "9.3",
                "category": "troubleshooting",
                "date": "2024-01-01",
            }
        ],
    )
    return AgentsConfig(
        chunks_path=chunks_path,
        checkpoint_path=tmp_path / "checkpoints.sqlite",
        mock_llm=True,
        classifier_model="claude-haiku-4-5-20251001",
        generator_model="claude-haiku-4-5-20251001",
        max_iterations=5,
        max_tokens=10000,
    )


def test_chitchat_routes_to_llm_only(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    store = MemoryCheckpointStore()

    payload = run_agents_core(
        config,
        query="Hello there!",
        checkpoint_store=store,
        retrieval_engine=FakeRetrievalEngine(_hits()),
    )

    assert payload["intent"] == "chitchat"
    assert payload["path_taken"] == ["classify_intent", "chitchat"]
    assert payload["answer"]


def test_factual_routes_through_retrieve_and_generate(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    store = MemoryCheckpointStore()
    retrieval = FakeRetrievalEngine(_hits())

    payload = run_agents_core(
        config,
        query="How do I reset the password?",
        checkpoint_store=store,
        retrieval_engine=retrieval,
    )

    assert retrieval.queries == ["How do I reset the password?"]
    assert payload["intent"] == "factual"
    assert payload["path_taken"] == ["classify_intent", "retrieve", "generate"]
    assert payload["retrieved_contexts"][0]["chunk_id"] == "C1"
    assert "Reset the queue manager password" in payload["answer"]


def test_escalate_routes_to_human(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    payload = run_agents_core(
        config,
        query="Delete the admin account right now.",
        checkpoint_store=MemoryCheckpointStore(),
        retrieval_engine=FakeRetrievalEngine(_hits()),
    )

    assert payload["intent"] == "escalate"
    assert payload["path_taken"] == ["classify_intent", "escalate"]
    assert "human review" in payload["answer"].lower()


def test_clarify_routes_when_request_is_underspecified(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    payload = run_agents_core(
        config,
        query="Help?",
        checkpoint_store=MemoryCheckpointStore(),
        retrieval_engine=FakeRetrievalEngine(_hits()),
    )

    assert payload["intent"] == "clarify"
    assert payload["path_taken"] == ["classify_intent", "clarify"]
    assert payload["clarification"]


def test_checkpoint_store_is_updated(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    store = MemoryCheckpointStore()

    payload = run_agents_core(
        config,
        query="Hello there!",
        checkpoint_store=store,
        retrieval_engine=FakeRetrievalEngine(_hits()),
        session_id="session-1",
    )

    assert store.load("session-1") is not None
    assert store.load("session-1")["answer"] == payload["answer"]
