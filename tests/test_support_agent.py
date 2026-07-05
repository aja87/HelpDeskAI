from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver

import scripts.run_agent as run_agent
from helpdeskai.agents.support_agent import AgentConfig, SupportAgent, build_support_graph
from helpdeskai.rag.models import RagContext, RagResult, StageTiming


class FakeClassifier:
    def __init__(
        self,
        intent: str = "technical_question",
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
            "technical_question": "answer_with_rag",
            "crm_question": "crm_support",
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


class FakeCrm:
    def __init__(self, *, fail: bool = False, customer_error: str | None = None) -> None:
        self.fail = fail
        self.customer_error = customer_error
        self.calls: list[tuple[str, str]] = []
        self.created: list[dict[str, str]] = []

    def get_customer(self, customer_id: str) -> dict:
        if self.fail:
            raise RuntimeError("crm down")
        self.calls.append(("get_customer", customer_id))
        if self.customer_error:
            return {"error": self.customer_error, "customer_id": customer_id}
        return {"customer_id": customer_id, "name": "Acme Europe"}

    def get_subscription_status(self, customer_id: str) -> dict:
        if self.fail:
            raise RuntimeError("crm down")
        self.calls.append(("get_subscription_status", customer_id))
        return {
            "customer_id": customer_id,
            "status": "active",
            "plan": "enterprise",
            "seats_used": 84,
            "seats_total": 120,
            "renewal_date": "2026-11-30",
        }

    def create_ticket(
        self,
        *,
        customer_id: str,
        subject: str,
        body: str,
        priority: str = "medium",
    ) -> dict:
        if self.fail:
            raise RuntimeError("crm down")
        ticket = {
            "ticket_id": "TCK-9999",
            "customer_id": customer_id,
            "subject": subject,
            "body": body,
            "priority": priority,
        }
        self.created.append(ticket)
        return {"success": True, "ticket": ticket}


def test_agent_routes_clear_support_question_through_rag() -> None:
    rag = FakeRag(calls=[])
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("technical_question"),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {"question": "How do I configure SAML login in NovaCloud?", "path_taken": []}
    )

    assert rag.calls == ["How do I configure SAML login in NovaCloud?"]
    assert state["answer"] == "Configure SAML from the admin console [chunk-1]"
    assert state["intent"] == "technical_question"
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
        intent_classifier=FakeClassifier("technical_question"),
        config=AgentConfig(max_iterations=1, max_session_tokens=10_000),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {"question": "How do I configure SAML login in NovaCloud?", "path_taken": []}
    )

    assert state["budget_exceeded"] is True
    assert "budget d'execution" in state["answer"]
    assert state["path_taken"] == ["classify_intent", "generate"]


def test_crm_question_budget_exceeded_does_not_require_account_context() -> None:
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question"),
        config=AgentConfig(max_iterations=1, max_session_tokens=10_000),
        crm_client=FakeCrm(),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de cust_acme ?", "path_taken": []})

    assert state["budget_exceeded"] is True
    assert state["account_context"] is None
    assert "budget d'execution" in state["answer"]


def test_sensitive_action_interrupts_and_resumes_after_approval() -> None:
    checkpointer = MemorySaver()
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question", sensitive=True),
        checkpointer=checkpointer,
        ticket_executor=lambda action: "Ticket TCK-0001 cree: Escalade support (high).",
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
        intent_classifier=FakeClassifier("technical_question"),
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
            return '{"intent":"crm_question","confidence":0.91,"sensitive":true}'

    classifier = LlmIntentClassifier(llm=FakeLlm())

    decision = classifier.classify("Deactivate this suspended account")

    assert decision.intent == "crm_question"
    assert decision.route == "sensitive_action"
    assert decision.confidence == 0.91
    assert decision.sensitive is True


def test_llm_classifier_accepts_json_object_inside_text() -> None:
    from helpdeskai.agents.support_agent import LlmIntentClassifier

    class FakeLlm:
        def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
            return '```json\n{"intent":"chitchat","confidence":0.83,"sensitive":false}\n```'

    classifier = LlmIntentClassifier(llm=FakeLlm())

    decision = classifier.classify("hello")

    assert decision.intent == "chitchat"
    assert decision.route == "chitchat"


def test_llm_classifier_prompt_covers_techqa_terms() -> None:
    from helpdeskai.agents.support_agent import LlmIntentClassifier

    assert "WebSphere Portal Server" in LlmIntentClassifier.prompt
    assert "wpcollector" in LlmIntentClassifier.prompt
    assert "hors support informatique/technique" in LlmIntentClassifier.prompt


def test_llm_classifier_rejects_unknown_intent() -> None:
    import pytest

    from helpdeskai.agents.support_agent import IntentClassificationError, LlmIntentClassifier

    class FakeLlm:
        def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
            return '{"intent":"account status","confidence":0.91,"sensitive":false}'

    classifier = LlmIntentClassifier(llm=FakeLlm())

    with pytest.raises(IntentClassificationError, match="unsupported intent"):
        classifier.classify("What is the status of my account?")


def test_crm_question_uses_crm_branch_without_rag() -> None:
    rag = FakeRag(calls=[])
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("crm_question"),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "What is the status of my account?", "path_taken": []})

    assert rag.calls == []
    assert state["route"] == "crm_support"
    assert state["path_taken"][-1] == "generate"


def test_crm_question_with_customer_id_calls_crm_before_answer() -> None:
    rag = FakeRag(calls=[])
    crm = FakeCrm()
    graph = build_support_graph(
        rag_pipeline=rag,
        intent_classifier=FakeClassifier("crm_question"),
        crm_client=crm,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de cust_acme ?", "path_taken": []})

    assert rag.calls == []
    assert crm.calls == [("get_customer", "cust_acme"), ("get_subscription_status", "cust_acme")]
    assert "active" in state["answer"]
    assert state["account_context"]["customer"]["name"] == "Acme Europe"


def test_crm_question_without_customer_id_asks_account_clarification() -> None:
    crm = FakeCrm()
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question"),
        crm_client=crm,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de mon compte ?", "path_taken": []})

    assert crm.calls == []
    assert "identifiant client" in state["answer"]


def test_crm_question_stops_after_customer_lookup_error() -> None:
    crm = FakeCrm(customer_error="rate_limited")
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question"),
        crm_client=crm,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de cust_acme ?", "path_taken": []})

    assert crm.calls == [("get_customer", "cust_acme")]
    assert "limite temporairement" in state["answer"]


def test_sensitive_approved_action_calls_crm_create_ticket() -> None:
    checkpointer = MemorySaver()
    crm = FakeCrm()
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question", sensitive=True),
        checkpointer=checkpointer,
        crm_client=crm,
    )
    config = {"configurable": {"thread_id": "account-mcp-1"}}

    graph.invoke(
        {"question": "Escalade le compte cust_acme pour acces admin bloque", "path_taken": []},
        config=config,
    )
    graph.update_state(config, {"approval": "approved"})
    final = graph.invoke(None, config=config)

    assert crm.created[0]["customer_id"] == "cust_acme"
    assert final["action_result"].startswith("Ticket TCK-9999 cree")


def test_sensitive_action_reuses_customer_id_from_crm_context() -> None:
    checkpointer = MemorySaver()
    crm = FakeCrm()
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question", sensitive=True),
        checkpointer=checkpointer,
        crm_client=crm,
    )
    config = {"configurable": {"thread_id": "account-mcp-context"}}

    state = graph.invoke(
        {
            "question": "je veux annuler notre souscription",
            "path_taken": [],
            "account_context": {"customer": {"customer_id": "cust_acme"}},
        },
        config=config,
    )

    assert state["pending_action"]["args"]["customer_id"] == "cust_acme"


def test_sensitive_rejected_action_does_not_call_crm() -> None:
    checkpointer = MemorySaver()
    crm = FakeCrm()
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question", sensitive=True),
        checkpointer=checkpointer,
        crm_client=crm,
    )
    config = {"configurable": {"thread_id": "account-mcp-2"}}

    graph.invoke(
        {"question": "Escalade le compte cust_acme pour acces admin bloque", "path_taken": []},
        config=config,
    )
    graph.update_state(config, {"approval": "rejected"})
    final = graph.invoke(None, config=config)

    assert crm.created == []
    assert "annulee" in final["answer"]


def test_mcp_failure_produces_graceful_answer() -> None:
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question"),
        crm_client=FakeCrm(fail=True),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de cust_acme ?", "path_taken": []})

    assert "CRM MCP est indisponible" in state["answer"]
    assert state["account_context"]["error"] == "mcp_unavailable"


def test_crm_generate_clears_stale_rag_sources() -> None:
    graph = build_support_graph(
        rag_pipeline=FakeRag(calls=[]),
        intent_classifier=FakeClassifier("crm_question"),
        crm_client=FakeCrm(),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {
            "question": "Quel est le statut de cust_acme ?",
            "path_taken": [],
            "sources": ["old-source"],
        }
    )

    assert state["sources"] == []


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
