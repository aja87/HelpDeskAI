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


class CrmClient(Protocol):
    """Minimal CRM MCP adapter contract consumed by the graph."""

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Return a CRM customer."""

    def get_subscription_status(self, customer_id: str) -> dict[str, Any]:
        """Return subscription status for a CRM customer."""

    def create_ticket(
        self,
        *,
        customer_id: str,
        subject: str,
        body: str,
        priority: str = "medium",
    ) -> dict[str, Any]:
        """Create a CRM support ticket."""


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
    account_context: dict[str, Any] | None
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
    {"technical_question", "crm_question", "out_of_scope", "chitchat", "ambiguous"}
)


class IntentClassificationError(ValueError):
    """Raised when the LLM does not return the expected intent schema."""


def _load_intent_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise IntentClassificationError("intent classifier JSON must be an object")
    return payload


def _route_for_intent(intent: str) -> str:
    if intent == "technical_question":
        return "answer_with_rag"
    if intent == "crm_question":
        return "crm_support"
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


def _extract_customer_id(question: str) -> str | None:
    match = re.search(r"\bcust_[a-z0-9_]{2,40}\b", question, flags=re.IGNORECASE)
    return match.group(0).lower() if match else None


class LlmIntentClassifier:
    """LLM classifier following the routing pattern used in the M06 exercises."""

    prompt = """Classe la question utilisateur en EXACTEMENT UNE intention metier :
- "technical_question" : question technique/documentaire qui peut etre traitee par la base de
  connaissance RAG, y compris logiciels d'entreprise, administration systeme, configuration,
  erreurs, API, WebSphere Portal Server, wpcollector, SAML, utilisateurs, roles ou acces
- "crm_question" : question specifique a un client/compte reel necessitant le CRM, par exemple
  statut de cust_xxx, abonnement d'un client, tickets recents ou compte suspendu
- "out_of_scope" : hors support informatique/technique et hors base de connaissance
- "chitchat" : salutation, politesse ou conversation courte
- "ambiguous" : demande trop vague pour repondre

Ajoute "sensitive": true uniquement si l'utilisateur demande une action sur le SI
comme creer/escalader un ticket, supprimer un compte, desactiver un utilisateur
ou modifier un abonnement. Sinon "sensitive": false.

Reponds uniquement avec JSON valide, sans markdown :
{{"intent":"technical_question|crm_question|out_of_scope|chitchat|ambiguous","confidence":0.0,"sensitive":false}}

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
            payload = _load_intent_json(raw)
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


def _customer_id_from_context(state: SupportAgentState) -> str | None:
    context = state.get("account_context") or {}
    customer = context.get("customer") if isinstance(context, dict) else None
    if isinstance(customer, dict):
        customer_id = customer.get("customer_id")
        return str(customer_id) if customer_id else None
    customer_id = context.get("customer_id") if isinstance(context, dict) else None
    return str(customer_id) if customer_id else None


def _make_action(question: str, *, customer_id: str | None = None) -> dict[str, Any]:
    subject = "Validation action support sensible"
    if re.search(r"\bescalad", question, flags=re.IGNORECASE):
        subject = "Escalade support"
    elif re.search(r"\bdelete\b|\bsupprime", question, flags=re.IGNORECASE):
        subject = "Validation suppression de donnee"
    return {
        "tool": "create_ticket",
        "args": {
            "customer_id": _extract_customer_id(question) or customer_id or "",
            "subject": subject,
            "body": question,
            "priority": "high",
        },
        "reason": "Action sensible demandee par l'utilisateur",
    }


def _format_mcp_error(prefix: str, exc: Exception) -> str:
    return f"{prefix} Le service CRM MCP est indisponible: {exc}"


def _crm_error_answer(error: str, customer_id: str) -> str:
    if error == "customer_not_found":
        return (
            f"Je n'ai pas trouve de compte CRM exploitable pour {customer_id}. "
            "Verifiez l'identifiant client."
        )
    if error == "rate_limited":
        return "Le CRM MCP limite temporairement les appels. Reessayez dans quelques instants."
    if error == "unauthorized":
        return "Je ne peux pas consulter le CRM MCP: le jeton d'authentification est invalide."
    return f"Le CRM MCP a retourne une erreur: {error}."


def _account_answer(
    question: str,
    crm_client: CrmClient | None,
) -> tuple[str, dict[str, Any] | None]:
    customer_id = _extract_customer_id(question)
    if customer_id is None:
        return (
            "Votre demande concerne un compte NovaCloud. Pouvez-vous fournir l'identifiant "
            "client au format cust_xxx pour que je consulte le CRM ?",
            None,
        )
    if crm_client is None:
        return (
            "Je ne peux pas consulter le CRM MCP pour le moment. Conservez l'identifiant "
            f"{customer_id} dans la demande et reessayez quand l'integration MCP est active.",
            {"customer_id": customer_id, "error": "mcp_not_configured"},
        )
    try:
        customer = crm_client.get_customer(customer_id)
        if customer.get("error"):
            return _crm_error_answer(customer["error"], customer_id), {"customer": customer}
        subscription = crm_client.get_subscription_status(customer_id)
    except Exception as exc:
        return (
            _format_mcp_error(
                f"Je n'ai pas pu recuperer le statut du compte {customer_id}.",
                exc,
            ),
            {"customer_id": customer_id, "error": "mcp_unavailable"},
        )
    context = {"customer": customer, "subscription": subscription}
    if subscription.get("error"):
        return _crm_error_answer(subscription["error"], customer_id), context
    answer = (
        f"Compte {customer.get('name', customer_id)} ({customer_id}) : abonnement "
        f"{subscription.get('status')} sur l'offre {subscription.get('plan')} "
        f"NovaCloud, {subscription.get('seats_used')}/{subscription.get('seats_total')} "
        f"sieges utilises. Renouvellement prevu le {subscription.get('renewal_date')}."
    )
    return answer, context


def build_support_graph(
    *,
    rag_pipeline: RagRunner | None = None,
    intent_classifier: IntentClassifier | None = None,
    config: AgentConfig = AgentConfig(),
    checkpointer: Any | None = None,
    interrupt_sensitive_actions: bool = True,
    ticket_executor: Any | None = None,
    crm_client: CrmClient | None = None,
) -> RunnableGraph:
    """Build the Phase 5 LangGraph agent."""
    rag = rag_pipeline or AdvancedRagPipeline()
    classifier = intent_classifier or LlmIntentClassifier()
    execute_ticket = ticket_executor

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
            result["pending_action"] = _make_action(
                question,
                customer_id=_customer_id_from_context(state),
            )
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
        account_context = None
        if state.get("budget_exceeded"):
            answer = state.get("answer", "")
        elif route == "crm_support":
            answer, account_context = _account_answer(state.get("question", ""), crm_client)
        elif route == "chitchat":
            answer = "Bonjour, je peux vous aider sur les questions support technique."
        elif route == "out_of_scope":
            answer = (
                "Cette demande sort du perimetre HelpDeskAI. Je peux traiter les questions "
                "support techniques documentees dans la base de connaissance."
            )
        else:
            answer = state.get("_rag_answer") or "Je n'ai pas trouve de reponse sourcee."
        update = _append_path(state, "generate", answer) | {"answer": answer}
        if route != "answer_with_rag":
            update["sources"] = []
        if route == "crm_support":
            update["account_context"] = account_context
        return update

    def clarification(state: SupportAgentState) -> dict[str, Any]:
        answer = (
            "Pouvez-vous preciser le produit concerne, le message d'erreur exact "
            "et l'action deja tentee ?"
        )
        return _append_path(state, "clarification", answer) | {
            "answer": answer,
            "sources": [],
        }

    def escalate(state: SupportAgentState) -> dict[str, Any]:
        action = state.get("pending_action") or _make_action(
            state.get("question", ""),
            customer_id=_customer_id_from_context(state),
        )
        approval = state.get("approval")
        if approval != "approved":
            answer = "Action sensible annulee ou en attente d'approbation."
            return _append_path(state, "escalate", action, answer) | {
                "pending_action": action,
                "answer": answer,
                "action_result": answer,
            }
        if crm_client is not None and action.get("tool") == "create_ticket":
            args = action["args"]
            if not args.get("customer_id"):
                answer = (
                    "Action sensible approuvee, mais aucun identifiant client cust_xxx n'est "
                    "present. Je conserve l'action en attente."
                )
                return _append_path(state, "escalate", action, answer) | {
                    "pending_action": action,
                    "action_result": "missing_customer_id",
                    "answer": answer,
                }
            try:
                payload = crm_client.create_ticket(
                    customer_id=args["customer_id"],
                    subject=args["subject"],
                    body=args["body"],
                    priority=args["priority"],
                )
            except Exception as exc:
                answer = _format_mcp_error(
                    "Action approuvee, mais le ticket CRM n'a pas ete cree.",
                    exc,
                )
                return _append_path(state, "escalate", action, answer) | {
                    "pending_action": action,
                    "action_result": "mcp_unavailable",
                    "answer": answer,
                }
            if payload.get("error"):
                answer = (
                    "Action approuvee, mais le CRM a refuse la creation du ticket: "
                    f"{payload.get('error')}."
                )
                return _append_path(state, "escalate", action, answer) | {
                    "pending_action": action,
                    "action_result": payload,
                    "answer": answer,
                }
            ticket = payload.get("ticket", payload)
            result = (
                f"Ticket {ticket.get('ticket_id')} cree dans le CRM: "
                f"{ticket.get('subject')} ({ticket.get('priority')})."
            )
        elif execute_ticket is not None:
            result = execute_ticket(action)
        else:
            answer = (
                "Action sensible approuvee, mais aucun client CRM MCP n'est configure. "
                "Je conserve l'action en attente."
            )
            return _append_path(state, "escalate", action, answer) | {
                "pending_action": action,
                "action_result": "mcp_not_configured",
                "answer": answer,
            }
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
        crm_client: CrmClient | None = None,
    ) -> SupportAgent:
        graph = build_support_graph(
            rag_pipeline=rag_pipeline,
            intent_classifier=intent_classifier,
            config=config,
            checkpointer=checkpointer,
            crm_client=crm_client,
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
