from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

import scripts.run_agent as run_agent
from helpdeskai.agents.support_agent import (
    AgentConfig,
    IntentDecision,
    SupportAgent,
    build_support_graph,
)


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

    def classify(self, question: str) -> IntentDecision:
        if self.sensitive or self.intent == "sensitive_action":
            return IntentDecision(
                "sensitive_action",
                self.confidence,
                sensitive=True,
            )
        if self.intent == "ambiguous":
            return IntentDecision("ambiguous", self.confidence, ambiguous=True)
        return IntentDecision(self.intent, self.confidence)


class FakeMcp:
    def __init__(self, *, fail: bool = False, customer_error: str | None = None) -> None:
        self.fail = fail
        self.customer_error = customer_error
        self.calls: list[tuple[str, str]] = []
        self.created: list[dict[str, str]] = []
        self.searches: list[dict[str, object]] = []

    def get_customer(self, customer_id: str) -> dict:
        if self.fail:
            raise RuntimeError("crm down")
        self.calls.append(("get_customer", customer_id))
        if self.customer_error:
            return {"error": self.customer_error, "customer_id": customer_id}
        return {
            "customer_id": customer_id,
            "name": "Acme Europe",
            "tenant": "acme-prod",
        }

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

    def search_knowledge(
        self,
        query: str,
        *,
        top_k: int = 5,
        product: str | None = None,
        version: str | None = None,
        tenant: str | None = None,
    ) -> dict:
        if self.fail:
            raise RuntimeError("knowledge down")
        self.searches.append(
            {
                "query": query,
                "top_k": top_k,
                "product": product,
                "version": version,
                "tenant": tenant,
            }
        )
        return {
            "query": query,
            "top_k": top_k,
            "filters": {"product": product, "version": version, "tenant": tenant},
            "results": [
                {
                    "source_id": "chunk-1",
                    "document_id": "doc-1",
                    "snippet": "Configure SAML from the admin console.",
                    "score": 0.9,
                    "metadata": {"product": "NovaCloud"},
                    "source_scores": {"hybrid": 0.9},
                }
            ],
        }


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
        self.prompts.append(prompt)
        return "Configure SAML from the admin console [chunk-1]"


def test_technical_question_plans_executes_knowledge_mcp_and_generates() -> None:
    mcp = FakeMcp()
    llm = FakeLlm()
    graph = build_support_graph(
        intent_classifier=FakeClassifier("technical_question"),
        mcp_client=mcp,
        llm=llm,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {"question": "How do I configure SAML login in NovaCloud?", "path_taken": []}
    )

    assert state["tool_plan"][0]["tool"] == "search_knowledge"
    assert mcp.searches[0]["query"] == "How do I configure SAML login in NovaCloud?"
    assert state["sources"] == ["chunk-1"]
    assert "[chunk-1]" in llm.prompts[0]
    assert state["answer"] == "Configure SAML from the admin console [chunk-1]"
    assert state["quality_passed"] is True
    assert state["path_taken"] == [
        "classify_intent",
        "plan_mcp_calls",
        "execute_mcp_calls",
        "generate_answer",
        "quality_check",
    ]


def test_account_question_chains_crm_mcp_calls() -> None:
    mcp = FakeMcp()
    graph = build_support_graph(
        intent_classifier=FakeClassifier("account_question"),
        mcp_client=mcp,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de cust_acme ?", "path_taken": []})

    assert [call["tool"] for call in state["tool_plan"]] == [
        "get_customer",
        "get_subscription_status",
    ]
    assert mcp.calls == [("get_customer", "cust_acme"), ("get_subscription_status", "cust_acme")]
    assert "active" in state["answer"]
    assert state["account_context"]["customer"]["name"] == "Acme Europe"


def test_account_plus_knowledge_chains_crm_then_knowledge_mcp() -> None:
    mcp = FakeMcp()
    llm = FakeLlm()
    graph = build_support_graph(
        intent_classifier=FakeClassifier("account_plus_knowledge_question"),
        mcp_client=mcp,
        llm=llm,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke(
        {
            "question": "Le client cust_acme peut-il configurer SAML ?",
            "path_taken": [],
        }
    )

    assert [call["tool"] for call in state["tool_plan"]] == [
        "get_customer",
        "get_subscription_status",
        "search_knowledge",
    ]
    assert mcp.calls == [("get_customer", "cust_acme"), ("get_subscription_status", "cust_acme")]
    assert mcp.searches[0]["tenant"] == "acme-prod"
    assert "Contexte CRM" in llm.prompts[0]
    assert state["sources"] == ["chunk-1"]


def test_agent_asks_for_clarification_when_question_is_ambiguous() -> None:
    graph = build_support_graph(
        intent_classifier=FakeClassifier("ambiguous", 0.4),
        mcp_client=FakeMcp(),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Erreur", "path_taken": []})

    assert state["tool_plan"] == []
    assert "Pouvez-vous preciser" in state["answer"]
    assert state["path_taken"] == ["classify_intent", "ask_clarification"]


def test_low_confidence_routes_to_clarification_without_planning_tools() -> None:
    graph = build_support_graph(
        intent_classifier=FakeClassifier("technical_question", 0.2),
        mcp_client=FakeMcp(),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "SAML", "path_taken": []})

    assert state["tool_plan"] == []
    assert state["path_taken"] == ["classify_intent", "ask_clarification"]


def test_chitchat_and_out_of_scope_do_not_plan_tools() -> None:
    graph = build_support_graph(
        intent_classifier=FakeClassifier("chitchat"),
        mcp_client=FakeMcp(),
        interrupt_sensitive_actions=False,
    )
    chitchat = graph.invoke({"question": "Bonjour", "path_taken": []})

    graph = build_support_graph(
        intent_classifier=FakeClassifier("out_of_scope"),
        mcp_client=FakeMcp(),
        interrupt_sensitive_actions=False,
    )
    out_of_scope = graph.invoke({"question": "Can you book a flight?", "path_taken": []})

    assert chitchat["tool_plan"] == []
    assert chitchat["path_taken"] == ["classify_intent", "direct_answer"]
    assert out_of_scope["tool_plan"] == []
    assert out_of_scope["path_taken"] == ["classify_intent", "escalate_to_human"]


def test_budget_exceeded_escalates_without_tool_planning() -> None:
    graph = build_support_graph(
        intent_classifier=FakeClassifier("technical_question"),
        config=AgentConfig(max_iterations=1, max_session_tokens=10_000),
        mcp_client=FakeMcp(),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "How do I configure SAML?", "path_taken": []})

    assert state["budget_exceeded"] is True
    assert state["tool_plan"] == []
    assert state["path_taken"] == ["classify_intent", "escalate_to_human"]


def test_account_question_without_customer_id_asks_specific_clarification() -> None:
    mcp = FakeMcp()
    graph = build_support_graph(
        intent_classifier=FakeClassifier("account_question"),
        mcp_client=mcp,
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Quel est le statut de mon compte ?", "path_taken": []})

    assert mcp.calls == []
    assert state["tool_plan"] == []
    assert "identifiant client" in state["answer"]
    assert state["path_taken"] == [
        "classify_intent",
        "plan_mcp_calls",
        "ask_clarification",
    ]


def test_sensitive_action_interrupts_after_human_approval_request_and_resumes() -> None:
    checkpointer = MemorySaver()
    mcp = FakeMcp()
    graph = build_support_graph(
        intent_classifier=FakeClassifier("sensitive_action", sensitive=True),
        checkpointer=checkpointer,
        mcp_client=mcp,
    )
    config = {"configurable": {"thread_id": "account-mcp-1"}}

    state = graph.invoke(
        {"question": "Escalade le compte cust_acme pour acces admin bloque", "path_taken": []},
        config=config,
    )

    assert state["pending_action"]["tool"] == "create_ticket"
    assert state["path_taken"] == [
        "classify_intent",
        "plan_mcp_calls",
        "request_human_approval",
    ]
    assert mcp.created == []

    graph.update_state(config, {"approval": "approved"})
    final = graph.invoke(None, config=config)

    assert mcp.created[0]["customer_id"] == "cust_acme"
    assert final["action_result"]["ticket"]["ticket_id"] == "TCK-9999"
    assert final["path_taken"][-2:] == ["generate_answer", "quality_check"]


def test_sensitive_rejected_action_does_not_call_mcp_mutation() -> None:
    checkpointer = MemorySaver()
    mcp = FakeMcp()
    graph = build_support_graph(
        intent_classifier=FakeClassifier("sensitive_action", sensitive=True),
        checkpointer=checkpointer,
        mcp_client=mcp,
    )
    config = {"configurable": {"thread_id": "account-mcp-2"}}

    graph.invoke(
        {"question": "Escalade le compte cust_acme pour acces admin bloque", "path_taken": []},
        config=config,
    )
    graph.update_state(config, {"approval": "rejected"})
    final = graph.invoke(None, config=config)

    assert mcp.created == []
    assert final["action_result"] == "approval_required"
    assert "annulee" in final["answer"]


def test_sensitive_action_without_customer_id_asks_clarification_before_approval() -> None:
    graph = build_support_graph(
        intent_classifier=FakeClassifier("sensitive_action", sensitive=True),
        mcp_client=FakeMcp(),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "Escalade ce probleme", "path_taken": []})

    assert state["tool_plan"] == []
    assert "cust_xxx" in state["answer"]
    assert "request_human_approval" not in state["path_taken"]


def test_mcp_failure_fails_quality_and_escalates() -> None:
    graph = build_support_graph(
        intent_classifier=FakeClassifier("technical_question"),
        mcp_client=FakeMcp(fail=True),
        interrupt_sensitive_actions=False,
    )

    state = graph.invoke({"question": "How do I configure SAML?", "path_taken": []})

    assert state["quality_passed"] is False
    assert state["quality_reason"] == "answer_not_reliable_enough"
    assert state["path_taken"][-1] == "escalate_to_human"
    assert "Knowledge MCP" in state["answer"]


def test_support_agent_wrapper_uses_stream_values_and_exports_mermaid() -> None:
    agent = SupportAgent.create(
        intent_classifier=FakeClassifier("account_question"),
        mcp_client=FakeMcp(),
    )

    state = agent.ask("Quel est le statut de cust_acme ?", thread_id="stream-values")
    mermaid = agent.draw_mermaid()

    assert state["path_taken"][-1] == "quality_check"
    assert "plan_mcp_calls" in mermaid
    assert "execute_mcp_calls" in mermaid


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
