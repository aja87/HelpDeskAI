"""LangGraph support agent with RAG, clarification, escalation and HIL."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from helpdeskai.rag.llm import ClaudeLlm, RagLlm
from helpdeskai.rag.models import RagResult
from helpdeskai.rag.pipeline import AdvancedRagPipeline


class RunnableGraph(Protocol):
    """Subset of the compiled LangGraph API used by the wrapper and tests."""

    def invoke(self, input: dict[str, Any] | None, config: dict[str, Any] | None = None) -> dict:
        """Run or resume a graph."""

    def update_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        """Patch a checkpointed state before resuming."""

    def get_graph(self) -> Any:
        """Return the drawable graph object."""


class RagRunner(Protocol):
    """Minimal RAG contract consumed by the agent."""

    def run(self, question: str) -> RagResult:
        """Answer a question with the RAG pipeline."""


class IntentClassifier(Protocol):
    """Minimal classifier contract consumed by the graph."""

    def classify(self, question: str) -> IntentDecision:
        """Classify one user question for graph routing."""


@dataclass(frozen=True)
class IntentDecision:
    """Classification result used for graph routing."""

    intent: str
    route: str
    confidence: float
    ambiguous: bool = False
    sensitive: bool = False


@dataclass(frozen=True)
class AgentConfig:
    """Runtime limits and thresholds for the support agent."""

    min_confidence: float = 0.62
    max_iterations: int = 5
    max_session_tokens: int = 10_000


class SupportAgentState(TypedDict, total=False):
    """State shared by all LangGraph nodes."""

    question: str
    intent: str
    route: str
    confidence: float
    ambiguous: bool
    sensitive: bool
    answer: str
    sources: list[str]
    pending_action: dict[str, Any] | None
    approval: str | None
    action_result: str | None
    iterations: int
    tokens_used: int
    budget_exceeded: bool
    path_taken: list[str]
    _rag_answer: str


def _estimate_tokens(*values: Any) -> int:
    """Cheap deterministic token budget estimate for offline execution."""
    text = " ".join(str(value) for value in values if value)
    return max(1, len(text) // 4)


def _append_path(state: SupportAgentState, node: str, *token_values: Any) -> dict[str, Any]:
    iterations = state.get("iterations", 0) + 1
    tokens_used = state.get("tokens_used", 0) + _estimate_tokens(*token_values)
    return {
        "iterations": iterations,
        "tokens_used": tokens_used,
        "budget_exceeded": state.get("budget_exceeded", False),
        "path_taken": state.get("path_taken", []) + [node],
    }


ALLOWED_INTENTS = frozenset(
    {"nova_question", "account_question", "out_of_scope", "chitchat", "ambiguous"}
)


class IntentClassificationError(ValueError):
    """Raised when the LLM does not return the expected intent schema."""


def _route_for_intent(intent: str) -> str:
    if intent == "nova_question":
        return "answer_with_rag"
    if intent == "account_question":
        return "account_support"
    if intent == "out_of_scope":
        return "out_of_scope"
    if intent == "chitchat":
        return "chitchat"
    if intent == "ambiguous":
        return "clarification"
    return "answer_with_rag"


def _decision_from_intent(
    intent: str,
    confidence: float | None = None,
    *,
    sensitive: bool = False,
) -> IntentDecision:
    route = "sensitive_action" if sensitive else _route_for_intent(intent)
    if sensitive:
        return IntentDecision(
            intent,
            route,
            confidence if confidence is not None else 0.86,
            sensitive=True,
        )
    if route == "clarification":
        return IntentDecision(
            intent,
            route,
            confidence if confidence is not None else 0.42,
            ambiguous=True,
        )
    if intent == "chitchat":
        return IntentDecision(intent, route, confidence if confidence is not None else 0.8)
    return IntentDecision(intent, route, confidence if confidence is not None else 0.78)


class LlmIntentClassifier:
    """LLM classifier following the routing pattern used in the M06 exercises."""

    prompt = """Classe la question utilisateur en EXACTEMENT UNE intention metier :
- "nova_question" : question sur NovaCloud, configuration, erreurs, API, SAML, usage produit
- "account_question" : question sur un compte, abonnement, droits, acces ou statut client
- "out_of_scope" : hors support NovaCloud
- "chitchat" : salutation, politesse ou conversation courte
- "ambiguous" : demande trop vague pour repondre

Ajoute "sensitive": true uniquement si l'utilisateur demande une action sur le SI
comme creer/escalader un ticket, supprimer un compte, desactiver un utilisateur
ou modifier un abonnement. Sinon "sensitive": false.

Reponds uniquement avec JSON valide, sans markdown :
{{"intent":"nova_question|account_question|out_of_scope|chitchat|ambiguous","confidence":0.0,"sensitive":false}}

Question : {question}"""

    def __init__(
        self,
        *,
        llm: RagLlm | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 80,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _llm(self) -> RagLlm:
        if self.llm is None:
            self.llm = ClaudeLlm()
        return self.llm

    def classify(self, question: str) -> IntentDecision:
        raw = self._llm().complete(
            self.prompt.format(question=question),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IntentClassificationError("intent classifier returned invalid JSON") from exc

        intent = payload.get("intent")
        if intent not in ALLOWED_INTENTS:
            raise IntentClassificationError(f"unsupported intent returned by LLM: {intent!r}")

        confidence = payload.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if confidence < 0 or confidence > 1:
                raise IntentClassificationError("intent confidence must be between 0 and 1")

        sensitive = payload.get("sensitive")
        if not isinstance(sensitive, bool):
            raise IntentClassificationError("intent sensitive flag must be a boolean")
        return _decision_from_intent(intent, confidence, sensitive=sensitive)


def _make_action(question: str) -> dict[str, Any]:
    subject = "Validation action support sensible"
    if re.search(r"\bescalad", question, flags=re.IGNORECASE):
        subject = "Escalade support"
    elif re.search(r"\bdelete\b|\bsupprime", question, flags=re.IGNORECASE):
        subject = "Validation suppression de donnee"
    return {
        "tool": "create_ticket",
        "args": {
            "customer_id": "unknown",
            "subject": subject,
            "body": question,
            "priority": "high",
        },
        "reason": "Action sensible demandee par l'utilisateur",
    }


def _execute_mock_ticket(action: dict[str, Any]) -> str:
    args = action["args"]
    return f"Ticket TCK-0001 cree: {args['subject']} ({args['priority']})."


def build_support_graph(
    *,
    rag_pipeline: RagRunner | None = None,
    intent_classifier: IntentClassifier | None = None,
    config: AgentConfig = AgentConfig(),
    checkpointer: Any | None = None,
    interrupt_sensitive_actions: bool = True,
    ticket_executor: Any | None = None,
) -> RunnableGraph:
    """Build the Phase 5 LangGraph agent."""
    rag = rag_pipeline or AdvancedRagPipeline()
    classifier = intent_classifier or LlmIntentClassifier()
    execute_ticket = ticket_executor or _execute_mock_ticket

    def classify_intent(state: SupportAgentState) -> dict[str, Any]:
        question = state.get("question", "")
        decision = classifier.classify(question)
        update = _append_path(state, "classify_intent", question, decision.intent, decision.route)
        exceeded = (
            update["iterations"] >= config.max_iterations
            or update["tokens_used"] >= config.max_session_tokens
        )
        update["budget_exceeded"] = exceeded
        result = update | {
            "intent": decision.intent,
            "route": decision.route,
            "confidence": decision.confidence,
            "ambiguous": decision.ambiguous,
            "sensitive": decision.sensitive,
        }
        if exceeded:
            result["answer"] = (
                "Je ne peux pas continuer dans le budget d'execution defini. "
                "Je transmets la demande a un agent humain avec le contexte disponible."
            )
        if decision.sensitive:
            result["pending_action"] = _make_action(question)
        return result

    def retrieve(state: SupportAgentState) -> dict[str, Any]:
        if state.get("route") != "answer_with_rag":
            return _append_path(state, "retrieve", state.get("route", "")) | {"sources": []}
        result = rag.run(state["question"])
        return _append_path(state, "retrieve", result.question_rewritten, result.sources) | {
            "sources": list(result.sources),
            "_rag_answer": result.answer,
        }

    def generate(state: SupportAgentState) -> dict[str, Any]:
        route = state.get("route")
        if state.get("budget_exceeded"):
            answer = state.get("answer", "")
        elif route == "account_support":
            answer = (
                "Votre demande concerne un compte NovaCloud. A ce stade, je peux la qualifier "
                "et conserver le contexte ; la consultation CRM sera branchee dans la phase MCP."
            )
        elif route == "chitchat":
            answer = "Bonjour, je peux vous aider sur les questions support NovaCloud."
        elif route == "out_of_scope":
            answer = (
                "Cette demande sort du perimetre HelpDeskAI. Je peux traiter les questions "
                "support liees a NovaCloud."
            )
        else:
            answer = state.get("_rag_answer") or "Je n'ai pas trouve de reponse sourcee."
        return _append_path(state, "generate", answer) | {"answer": answer}

    def clarification(state: SupportAgentState) -> dict[str, Any]:
        answer = (
            "Pouvez-vous preciser le produit NovaCloud concerne, le message d'erreur exact "
            "et l'action deja tentee ?"
        )
        return _append_path(state, "clarification", answer) | {
            "answer": answer,
            "sources": [],
        }

    def escalate(state: SupportAgentState) -> dict[str, Any]:
        action = state.get("pending_action") or _make_action(state.get("question", ""))
        approval = state.get("approval")
        if approval != "approved":
            answer = "Action sensible annulee ou en attente d'approbation."
            return _append_path(state, "escalate", action, answer) | {
                "pending_action": action,
                "answer": answer,
                "action_result": answer,
            }
        result = execute_ticket(action)
        return _append_path(state, "escalate", action) | {
            "pending_action": action,
            "action_result": result,
            "answer": result,
        }

    def route_after_classification(state: SupportAgentState) -> str:
        if state.get("budget_exceeded"):
            return "generate"
        if state.get("ambiguous") or state.get("confidence", 0.0) < config.min_confidence:
            return "clarification"
        if state.get("sensitive") or state.get("route") == "sensitive_action":
            return "escalate"
        if state.get("route") == "answer_with_rag":
            return "retrieve"
        return "generate"

    graph = StateGraph(SupportAgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("clarification", clarification)
    graph.add_node("escalate", escalate)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "clarification": "clarification",
            "escalate": "escalate",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("clarification", END)
    graph.add_edge("escalate", END)

    interrupt_before = ["escalate"] if interrupt_sensitive_actions else None
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


@contextmanager
def open_sqlite_checkpointer(path: str | Path):
    """Open a SQLite LangGraph checkpointer and close its connection."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(connection)
    finally:
        connection.close()


class SupportAgent:
    """Small convenience wrapper around the compiled LangGraph agent."""

    def __init__(self, graph: RunnableGraph) -> None:
        self.graph = graph

    @classmethod
    def create(
        cls,
        *,
        rag_pipeline: RagRunner | None = None,
        intent_classifier: IntentClassifier | None = None,
        config: AgentConfig = AgentConfig(),
        checkpointer: Any | None = None,
    ) -> SupportAgent:
        graph = build_support_graph(
            rag_pipeline=rag_pipeline,
            intent_classifier=intent_classifier,
            config=config,
            checkpointer=checkpointer,
        )
        return cls(graph)

    def ask(self, question: str, *, thread_id: str = "default") -> dict[str, Any]:
        state: SupportAgentState = {
            "question": question,
            "approval": None,
            "pending_action": None,
            "iterations": 0,
            "tokens_used": 0,
            "path_taken": [],
        }
        return self.graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

    def approve(self, *, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {"approval": "approved"})
        return self.graph.invoke(None, config=config)

    def reject(self, *, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {"approval": "rejected"})
        return self.graph.invoke(None, config=config)

    def draw_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def run_many(
        self,
        questions: Sequence[str],
        *,
        thread_prefix: str = "demo",
    ) -> list[dict[str, Any]]:
        return [
            self.ask(question, thread_id=f"{thread_prefix}-{index}")
            for index, question in enumerate(questions)
        ]
