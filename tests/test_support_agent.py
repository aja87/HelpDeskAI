from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver

import scripts.run_agent as run_agent
from helpdeskai.agents.support_agent import AgentConfig, SupportAgent, build_support_graph
from helpdeskai.rag.models import RagContext, RagResult, StageTiming


class FakeClassifier:
    def __init__(
        self,
        intent: str = "nova_question",
        confidence: float = 0.9,
        *,
        sensitive: bool = False,
    ) -> None:
        self.intent = intent
        self.confidence = confidence
        self.sensitive = sensitive

    def classify(self, question: str):
        from helpdeskai.agents.support_agent import IntentDecision

        if self.sensitive:
            return IntentDecision(self.intent, "sensitive_action", self.confidence, sensitive=True)
        if self.intent == "ambiguous":
            return IntentDecision(self.intent, "clarification", self.confidence, ambiguous=True)
        route = {
            "nova_question": "answer_with_rag",
            "account_question": "account_support",
            "out_of_scope": "out_of_scope",
            "chitchat": "chitchat",
        }[self.intent]
        return IntentDecision(self.intent, route, self.confidence)


@dataclass
class FakeRag:
    calls: list[str]

    def run(self, question: str) -> RagResult:
        self.calls.append(question)
        return RagResult(
            question_original=question,
            question_rewritten=f"rewritten {question}",
            answer="Configure SAML from the admin console [chunk-1]",
            contexts=[
                RagContext(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="SAML setup",
                    score=0.9,
                )
            ],
            sources=["chunk-1"],
            timings=[StageTiming("fake", 1.0)],
            model_names={"generator": "fake", "reranker": "fake"},
            prompt_version="strict",
            retrieval_mode="hybrid",
        )


def test_agent_routes_clear_support_question_through_rag() -> None:
    rag = FakeRag(calls=[])
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("nova_question"),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {"question": "How do I configure SAML login in NovaCloud?", "path_taken": []}
    )

    assert rag.calls == ["How do I configure SAML login in NovaCloud?"]
    assert state["answer"] == "Configure SAML from the admin console [chunk-1]"
    assert state["intent"] == "nova_question"
    assert state["route"] == "answer_with_rag"
    assert state["sources"] == ["chunk-1"]
    assert state["path_taken"] == ["classify_intent", "retrieve", "generate"]


def test_agent_asks_for_clarification_when_question_is_ambiguous() -> None:
    rag = FakeRag(calls=[])
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("ambiguous", 0.4),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Erreur", "path_taken": []})

    assert rag.calls == []
    assert "Pouvez-vous preciser" in state["answer"]
    assert state["path_taken"][-1] == "clarification"


def test_agent_falls_back_when_iteration_budget_is_exceeded() -> None:
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("nova_question"),
        config=AgentConfig(max_iterations=1, max_session_tokens=10_000),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {"question": "How do I configure SAML login in NovaCloud?", "path_taken": []}
    )

    assert state["budget_exceeded"] is True
    assert "budget d'execution" in state["answer"]
    assert state["path_taken"] == ["classify_intent", "generate"]


def test_sensitive_action_interrupts_and_resumes_after_approval() -> None:
    checkpointer = MemorySaver()
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("account_question", sensitive=True),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "account-1"}}

    state = graph.invoke(
        {"question": "Deactivate this suspended account", "path_taken": []},
        config=config,
    )

    assert state["pending_action"]["tool"] == "create_ticket"
    assert state["path_taken"][-1] == "classify_intent"

    graph.update_state(config, {"approval": "approved"})
    final = graph.invoke(None, config=config)

    assert final["action_result"].startswith("Ticket TCK-0001 cree")
    assert final["path_taken"][-1] == "escalate"


def test_support_agent_wrapper_exports_mermaid() -> None:
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("nova_question"),
        interrupt_sensitive_actions=False,
    )
    agent = SupportAgent(graph)

    mermaid = agent.draw_mermaid()

    assert "classify_intent" in mermaid
    assert "escalate" in mermaid


def test_llm_classifier_uses_exact_json_response() -> None:
    from helpdeskai.agents.support_agent import LlmIntentClassifier

    class FakeLlm:
        def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
            assert "Classe la question utilisateur" in prompt
            return '{"intent":"account_question","confidence":0.91,"sensitive":true}'

    classifier = LlmIntentClassifier(llm=FakeLlm())

    decision = classifier.classify("Deactivate this suspended account")

    assert decision.intent == "account_question"
    assert decision.route == "sensitive_action"
    assert decision.confidence == 0.91
    assert decision.sensitive is True


def test_llm_classifier_rejects_unknown_intent() -> None:
    import pytest

    from helpdeskai.agents.support_agent import IntentClassificationError, LlmIntentClassifier

    class FakeLlm:
        def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
            return '{"intent":"account status","confidence":0.91,"sensitive":false}'

    classifier = LlmIntentClassifier(llm=FakeLlm())

    with pytest.raises(IntentClassificationError, match="unsupported intent"):
        classifier.classify("What is the status of my account?")


def test_account_question_uses_account_branch_without_rag() -> None:
    rag = FakeRag(calls=[])
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("account_question"),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "What is the status of my account?", "path_taken": []})

    assert rag.calls == []
    assert state["route"] == "account_support"
    assert state["path_taken"][-1] == "generate"


def test_chitchat_and_out_of_scope_are_direct_responses() -> None:
    rag = FakeRag(calls=[])
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("chitchat"),
        interrupt_sensitive_actions=False,
    )

    chitchat = graph.invoke({"question": "Bonjour", "path_taken": []})

    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("out_of_scope"),
        interrupt_sensitive_actions=False,
    )
    out_of_scope = graph.invoke({"question": "Can you book a flight?", "path_taken": []})

    assert rag.calls == []
    assert chitchat["path_taken"][-1] == "generate"
    assert out_of_scope["path_taken"][-1] == "generate"


def test_run_agent_export_only_does_not_invoke_agent(monkeypatch, tmp_path) -> None:
    class FakeAgent:
        def draw_mermaid(self) -> str:
            return "graph TD;\n"

        def ask(self, question: str, *, thread_id: str = "default"):
            raise AssertionError("ask should not be called for export-only")

    class FakeCheckpointer:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(run_agent, "open_sqlite_checkpointer", lambda path: FakeCheckpointer())
    monkeypatch.setattr(run_agent.SupportAgent, "create", lambda **kwargs: FakeAgent())

    output = tmp_path / "agent_graph.mmd"

    assert run_agent.main(["--export-mermaid", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == "graph TD;\n"
