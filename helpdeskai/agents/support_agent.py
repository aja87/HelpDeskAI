"""LangGraph support agent with explicit MCP tool orchestration."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from helpdeskai.rag.llm import ClaudeLlm, RagLlm
from helpdeskai.rag.models import RagConfig
from helpdeskai.rag.prompts import get_prompt_variant

ToolName = Literal[
    "search_knowledge",
    "get_customer",
    "get_subscription_status",
    "create_ticket",
]


class ToolCall(TypedDict, total=False):
    """One planned MCP tool call."""

    tool: ToolName
    args: dict[str, Any]
    requires_approval: bool
    purpose: str


class ToolResult(TypedDict, total=False):
    """One executed MCP tool result."""

    tool: ToolName
    args: dict[str, Any]
    result: dict[str, Any]
    error: str | None


class RunnableGraph(Protocol):
    """Subset of the compiled LangGraph API used by the wrapper and tests."""

    def invoke(self, input: dict[str, Any] | None, config: dict[str, Any] | None = None) -> dict:
        """Run or resume a graph."""

    def stream(
        self,
        input: dict[str, Any] | None,
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str = "values",
    ) -> Any:
        """Stream graph states."""

    def update_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        """Patch a checkpointed state before resuming."""

    def get_graph(self) -> Any:
        """Return the drawable graph object."""


class IntentClassifier(Protocol):
    """Minimal classifier contract consumed by the graph."""

    def classify(self, question: str) -> IntentDecision:
        """Classify one user question for graph routing."""


class SupportMcpClient(Protocol):
    """MCP tools consumed by the support graph."""

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

    def search_knowledge(
        self,
        query: str,
        *,
        top_k: int = 5,
        product: str | None = None,
        version: str | None = None,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """Search the knowledge base through MCP."""


@dataclass(frozen=True)
class IntentDecision:
    """Classification result used for graph routing."""

    intent: str
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
    action_result: Any
    account_context: dict[str, Any] | None
    knowledge_contexts: list[dict[str, Any]]
    tool_plan: list[ToolCall]
    tool_results: list[ToolResult]
    needs_approval: bool
    clarification_needed: bool
    quality_passed: bool
    quality_reason: str
    iterations: int
    tokens_used: int
    budget_exceeded: bool
    path_taken: list[str]


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
    {
        "technical_question",
        "account_question",
        "account_plus_knowledge_question",
        "sensitive_action",
        "out_of_scope",
        "chitchat",
        "ambiguous",
    }
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
    if intent == "ambiguous":
        return "clarification"
    if intent == "chitchat":
        return "direct_answer"
    if intent == "out_of_scope":
        return "escalate_to_human"
    if intent == "sensitive_action":
        return "tool_backed"
    return "tool_backed"


def _decision_from_intent(
    intent: str,
    confidence: float | None = None,
    *,
    sensitive: bool = False,
) -> IntentDecision:
    normalized = intent
    if sensitive:
        normalized = "sensitive_action"
    if normalized == "ambiguous":
        return IntentDecision(
            normalized,
            confidence if confidence is not None else 0.42,
            ambiguous=True,
        )
    return IntentDecision(
        normalized,
        confidence if confidence is not None else 0.86 if sensitive else 0.78,
        sensitive=sensitive or normalized == "sensitive_action",
    )


def _extract_customer_id(question: str) -> str | None:
    match = re.search(r"\bcust_[a-z0-9_]{2,40}\b", question, flags=re.IGNORECASE)
    return match.group(0).lower() if match else None


class LlmIntentClassifier:
    """LLM classifier following the routing pattern used in the M06 exercises."""

    prompt = """Classe la question utilisateur en EXACTEMENT UNE intention metier :
- "technical_question" : question technique/documentaire traitee avec la base de connaissance
- "account_question" : question specifique a un client/compte necessitant le CRM
- "account_plus_knowledge_question" : question qui necessite a la fois CRM et documentation
- "out_of_scope" : hors support informatique/technique et hors base de connaissance
- "chitchat" : salutation, politesse ou conversation courte
- "ambiguous" : demande trop vague pour repondre

Ajoute "sensitive": true uniquement si l'utilisateur demande une action sur le SI
comme creer/escalader un ticket, supprimer un compte, desactiver un utilisateur
ou modifier un abonnement. Sinon "sensitive": false.

Reponds uniquement avec JSON valide, sans markdown :
{{"intent":"technical_question|account_question|account_plus_knowledge_question|out_of_scope|chitchat|ambiguous","confidence":0.0,"sensitive":false}}

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


def _make_ticket_call(question: str, *, customer_id: str | None = None) -> ToolCall:
    subject = "Validation action support sensible"
    if re.search(r"\bescalad|urgent", question, flags=re.IGNORECASE):
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
        "requires_approval": True,
        "purpose": "Creer un ticket support apres validation humaine.",
    }


def _format_mcp_error(prefix: str, exc: Exception) -> str:
    return f"{prefix} Le service MCP est indisponible: {exc}"


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


def _tool_result(
    tool: ToolName,
    args: dict[str, Any],
    result: dict[str, Any],
    error: str | None = None,
) -> ToolResult:
    return {"tool": tool, "args": args, "result": result, "error": error}


def _knowledge_contexts_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(item.get("source_id", "")),
            "document_id": str(item.get("document_id", "")),
            "content": str(item.get("snippet", "")),
            "score": float(item.get("score", 0.0)),
            "metadata": dict(item.get("metadata") or {}),
            "source_scores": dict(item.get("source_scores") or {}),
        }
        for item in result.get("results", [])
    ]


def _format_account_answer(context: dict[str, Any]) -> str:
    customer = context.get("customer") or {}
    subscription = context.get("subscription") or {}
    customer_id = customer.get("customer_id") or subscription.get("customer_id") or "client"
    if customer.get("error"):
        return _crm_error_answer(str(customer["error"]), str(customer_id))
    if subscription.get("error"):
        return _crm_error_answer(str(subscription["error"]), str(customer_id))
    return (
        f"Compte {customer.get('name', customer_id)} ({customer_id}) : abonnement "
        f"{subscription.get('status')} sur l'offre {subscription.get('plan')} "
        f"NovaCloud, {subscription.get('seats_used')}/{subscription.get('seats_total')} "
        f"sieges utilises. Renouvellement prevu le {subscription.get('renewal_date')}."
    )


def _run_graph_values(
    graph: RunnableGraph,
    input_state: dict[str, Any] | None,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a graph with stream_mode='values' and return the last state."""
    last: dict[str, Any] | None = None
    for event in graph.stream(input_state, config=config, stream_mode="values"):
        last = event
    return last or {}


def build_support_graph(
    *,
    intent_classifier: IntentClassifier | None = None,
    config: AgentConfig = AgentConfig(),
    rag_config: RagConfig = RagConfig(),
    llm: RagLlm | None = None,
    checkpointer: Any | None = None,
    interrupt_sensitive_actions: bool = True,
    mcp_client: SupportMcpClient | None = None,
) -> RunnableGraph:
    """Build the LangGraph support agent with explicit MCP orchestration."""
    if mcp_client is None:
        from helpdeskai.mcp_servers.client import StdioMcpClient

        mcp_client = StdioMcpClient()
    classifier = intent_classifier or LlmIntentClassifier()

    def _llm() -> RagLlm:
        nonlocal llm
        if llm is None:
            llm = ClaudeLlm()
        return llm

    def classify_intent(state: SupportAgentState) -> dict[str, Any]:
        question = state.get("question", "")
        decision = classifier.classify(question)
        intent = "sensitive_action" if decision.sensitive else decision.intent
        route = _route_for_intent(intent)
        update = _append_path(state, "classify_intent", question, intent, route)
        exceeded = (
            update["iterations"] >= config.max_iterations
            or update["tokens_used"] >= config.max_session_tokens
        )
        update["budget_exceeded"] = exceeded
        result = update | {
            "intent": intent,
            "route": route,
            "confidence": decision.confidence,
            "ambiguous": decision.ambiguous,
            "sensitive": decision.sensitive or intent == "sensitive_action",
            "tool_plan": [],
            "tool_results": [],
            "sources": [],
            "knowledge_contexts": [],
            "needs_approval": False,
            "clarification_needed": False,
        }
        if exceeded:
            result["answer"] = (
                "Je ne peux pas continuer dans le budget d'execution defini. "
                "Je transmets la demande a un agent humain avec le contexte disponible."
            )
        return result

    def ask_clarification(state: SupportAgentState) -> dict[str, Any]:
        answer = state.get("answer") or (
            "Pouvez-vous preciser le produit concerne, le message d'erreur exact "
            "et l'action deja tentee ?"
        )
        return _append_path(state, "ask_clarification", answer) | {
            "answer": answer,
            "sources": [],
        }

    def direct_answer(state: SupportAgentState) -> dict[str, Any]:
        if state.get("route") == "direct_answer" or state.get("intent") == "chitchat":
            answer = "Bonjour, je peux vous aider sur les questions support technique."
        else:
            answer = state.get("answer", "")
        return _append_path(state, "direct_answer", answer) | {
            "answer": answer,
            "sources": [],
        }

    def escalate_to_human(state: SupportAgentState) -> dict[str, Any]:
        answer = state.get("answer") or (
            "Je transmets cette demande a un agent humain avec le contexte disponible."
        )
        return _append_path(state, "escalate_to_human", answer) | {
            "answer": answer,
            "quality_passed": False,
        }

    def plan_mcp_calls(state: SupportAgentState) -> dict[str, Any]:
        question = state.get("question", "")
        intent = state.get("intent", "")
        customer_id = _extract_customer_id(question) or _customer_id_from_context(state)
        plan: list[ToolCall] = []
        update: dict[str, Any] = _append_path(state, "plan_mcp_calls", question, intent)

        if state.get("sensitive") or intent == "sensitive_action":
            ticket_call = _make_ticket_call(question, customer_id=customer_id)
            if not ticket_call["args"].get("customer_id"):
                return update | {
                    "clarification_needed": True,
                    "answer": (
                        "Cette action est sensible. Pouvez-vous fournir l'identifiant client "
                        "au format cust_xxx avant validation ?"
                    ),
                    "tool_plan": [],
                    "needs_approval": False,
                }
            plan.append(ticket_call)
        elif intent == "technical_question":
            plan.append(
                {
                    "tool": "search_knowledge",
                    "args": {"query": question, "top_k": rag_config.final_k},
                    "requires_approval": False,
                    "purpose": "Rechercher les sources documentaires pour repondre.",
                }
            )
        elif intent == "account_question":
            if not customer_id:
                return update | {
                    "clarification_needed": True,
                    "answer": (
                        "Votre demande concerne un compte NovaCloud. Pouvez-vous fournir "
                        "l'identifiant client au format cust_xxx ?"
                    ),
                    "tool_plan": [],
                    "needs_approval": False,
                }
            plan.extend(
                [
                    {
                        "tool": "get_customer",
                        "args": {"customer_id": customer_id},
                        "requires_approval": False,
                        "purpose": "Verifier l'identite CRM du client.",
                    },
                    {
                        "tool": "get_subscription_status",
                        "args": {"customer_id": customer_id},
                        "requires_approval": False,
                        "purpose": "Verifier le statut d'abonnement du client.",
                    },
                ]
            )
        elif intent == "account_plus_knowledge_question":
            if not customer_id:
                return update | {
                    "clarification_needed": True,
                    "answer": (
                        "Cette demande necessite le CRM et la documentation. Pouvez-vous "
                        "fournir l'identifiant client au format cust_xxx ?"
                    ),
                    "tool_plan": [],
                    "needs_approval": False,
                }
            plan.extend(
                [
                    {
                        "tool": "get_customer",
                        "args": {"customer_id": customer_id},
                        "requires_approval": False,
                        "purpose": "Verifier l'identite CRM du client.",
                    },
                    {
                        "tool": "get_subscription_status",
                        "args": {"customer_id": customer_id},
                        "requires_approval": False,
                        "purpose": "Verifier le statut d'abonnement du client.",
                    },
                    {
                        "tool": "search_knowledge",
                        "args": {"query": question, "top_k": rag_config.final_k},
                        "requires_approval": False,
                        "purpose": "Rechercher les sources documentaires pertinentes.",
                    },
                ]
            )

        needs_approval = any(call.get("requires_approval", False) for call in plan)
        pending_action = next((call for call in plan if call.get("requires_approval")), None)
        return update | {
            "tool_plan": plan,
            "needs_approval": needs_approval,
            "pending_action": pending_action,
            "clarification_needed": False,
        }

    def request_human_approval(state: SupportAgentState) -> dict[str, Any]:
        action = state.get("pending_action")
        answer = "Action sensible en attente d'approbation humaine."
        return _append_path(state, "request_human_approval", action) | {
            "answer": answer,
            "pending_action": action,
        }

    def execute_mcp_calls(state: SupportAgentState) -> dict[str, Any]:
        if state.get("needs_approval") and state.get("approval") != "approved":
            answer = "Action sensible annulee ou en attente d'approbation."
            return _append_path(state, "execute_mcp_calls", answer) | {
                "answer": answer,
                "action_result": "approval_required",
                "tool_results": [],
            }

        results: list[ToolResult] = []
        account_context: dict[str, Any] = {}
        knowledge_contexts: list[dict[str, Any]] = []
        sources: list[str] = []
        action_result: Any = None

        for call in state.get("tool_plan", []):
            tool = call["tool"]
            args = dict(call.get("args", {}))
            try:
                if tool == "get_customer":
                    payload = mcp_client.get_customer(str(args["customer_id"]))
                    account_context["customer"] = payload
                elif tool == "get_subscription_status":
                    payload = mcp_client.get_subscription_status(str(args["customer_id"]))
                    account_context["subscription"] = payload
                elif tool == "search_knowledge":
                    customer = account_context.get("customer") or {}
                    if args.get("tenant") is None and customer.get("tenant"):
                        args["tenant"] = customer["tenant"]
                    payload = mcp_client.search_knowledge(**args)
                    knowledge_contexts = _knowledge_contexts_from_result(payload)
                    sources = [context["chunk_id"] for context in knowledge_contexts]
                elif tool == "create_ticket":
                    payload = mcp_client.create_ticket(**args)
                    action_result = payload
                else:
                    payload = {"error": "unsupported_tool"}
                error = (
                    str(payload["error"])
                    if isinstance(payload, dict) and payload.get("error")
                    else None
                )
                results.append(_tool_result(tool, args, payload, error))
            except Exception as exc:
                payload = {"error": "mcp_unavailable", "details": str(exc)}
                results.append(_tool_result(tool, args, payload, "mcp_unavailable"))

        update = _append_path(state, "execute_mcp_calls", results) | {
            "tool_results": results,
            "account_context": account_context or None,
            "knowledge_contexts": knowledge_contexts,
            "sources": sources,
            "action_result": action_result,
        }
        return update

    def generate_answer(state: SupportAgentState) -> dict[str, Any]:
        if state.get("action_result") == "approval_required" and state.get("answer"):
            return _append_path(state, "generate_answer", state["answer"]) | {
                "answer": state["answer"]
            }
        tool_results = state.get("tool_results", [])
        failed = next((result for result in tool_results if result.get("error")), None)
        if failed:
            answer = _format_failed_tool_answer(failed)
        elif any(result["tool"] == "create_ticket" for result in tool_results):
            answer = _format_ticket_answer(state.get("action_result"))
        elif state.get("account_context") and state.get("knowledge_contexts"):
            answer = _generate_grounded_answer(
                question=state.get("question", ""),
                contexts=state.get("knowledge_contexts", []),
                account_context=state.get("account_context"),
                llm=_llm(),
                rag_config=rag_config,
            )
        elif state.get("knowledge_contexts"):
            answer = _generate_grounded_answer(
                question=state.get("question", ""),
                contexts=state.get("knowledge_contexts", []),
                account_context=None,
                llm=_llm(),
                rag_config=rag_config,
            )
        elif state.get("account_context"):
            answer = _format_account_answer(state["account_context"] or {})
        else:
            answer = "Je n'ai pas trouve de reponse sourcee exploitable."
        return _append_path(state, "generate_answer", answer) | {"answer": answer}

    def quality_check(state: SupportAgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        needs_sources = any(
            call["tool"] == "search_knowledge" for call in state.get("tool_plan", [])
        )
        has_error = any(result.get("error") for result in state.get("tool_results", []))
        passed = bool(answer) and not has_error
        if needs_sources:
            passed = passed and bool(state.get("sources"))
        reason = "ok" if passed else "answer_not_reliable_enough"
        return _append_path(state, "quality_check", reason) | {
            "quality_passed": passed,
            "quality_reason": reason,
        }

    def route_after_classification(state: SupportAgentState) -> str:
        if state.get("budget_exceeded"):
            return "escalate_to_human"
        if state.get("ambiguous") or state.get("confidence", 0.0) < config.min_confidence:
            return "ask_clarification"
        route = state.get("route")
        if route == "direct_answer":
            return "direct_answer"
        if route == "escalate_to_human":
            return "escalate_to_human"
        return "plan_mcp_calls"

    def route_after_plan(state: SupportAgentState) -> str:
        if state.get("clarification_needed"):
            return "ask_clarification"
        if state.get("needs_approval"):
            return "request_human_approval"
        return "execute_mcp_calls"

    def route_after_quality(state: SupportAgentState) -> str:
        return END if state.get("quality_passed") else "escalate_to_human"

    graph = StateGraph(SupportAgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("escalate_to_human", escalate_to_human)
    graph.add_node("plan_mcp_calls", plan_mcp_calls)
    graph.add_node("request_human_approval", request_human_approval)
    graph.add_node("execute_mcp_calls", execute_mcp_calls)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("quality_check", quality_check)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "ask_clarification": "ask_clarification",
            "direct_answer": "direct_answer",
            "escalate_to_human": "escalate_to_human",
            "plan_mcp_calls": "plan_mcp_calls",
        },
    )
    graph.add_conditional_edges(
        "plan_mcp_calls",
        route_after_plan,
        {
            "ask_clarification": "ask_clarification",
            "request_human_approval": "request_human_approval",
            "execute_mcp_calls": "execute_mcp_calls",
        },
    )
    graph.add_edge("request_human_approval", "execute_mcp_calls")
    graph.add_edge("execute_mcp_calls", "generate_answer")
    graph.add_edge("generate_answer", "quality_check")
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {END: END, "escalate_to_human": "escalate_to_human"},
    )
    graph.add_edge("ask_clarification", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("escalate_to_human", END)

    interrupt_after = ["request_human_approval"] if interrupt_sensitive_actions else None
    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def _format_failed_tool_answer(result: ToolResult) -> str:
    tool = result.get("tool", "outil")
    payload = result.get("result") or {}
    error = payload.get("error") or result.get("error")
    if tool in {"get_customer", "get_subscription_status", "create_ticket"}:
        return f"Le CRM MCP a retourne une erreur: {error}."
    return f"Le Knowledge MCP a retourne une erreur: {error}."


def _format_ticket_answer(action_result: Any) -> str:
    if not isinstance(action_result, dict):
        return str(action_result or "Action executee.")
    if action_result.get("error"):
        return (
            "Action approuvee, mais le CRM a refuse la creation du ticket: "
            f"{action_result['error']}."
        )
    ticket = action_result.get("ticket", action_result)
    return (
        f"Ticket {ticket.get('ticket_id')} cree dans le CRM: "
        f"{ticket.get('subject')} ({ticket.get('priority')})."
    )


def _generate_grounded_answer(
    *,
    question: str,
    contexts: list[dict[str, Any]],
    account_context: dict[str, Any] | None,
    llm: RagLlm,
    rag_config: RagConfig,
) -> str:
    if not contexts:
        return "Information non disponible dans les documents fournis."
    context_text = "\n\n".join(
        f"[{context['chunk_id']}] {context['content']}" for context in contexts
    )
    if account_context:
        crm_context = json.dumps(account_context, ensure_ascii=False)
        context_text = f"Contexte CRM:\n{crm_context}\n\n{context_text}"
    prompt = get_prompt_variant(rag_config.prompt_version)(question, context_text)
    return llm.complete(
        prompt,
        model=rag_config.generator_model,
        max_tokens=rag_config.max_generation_tokens,
        temperature=rag_config.temperature,
    )


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
        intent_classifier: IntentClassifier | None = None,
        config: AgentConfig = AgentConfig(),
        rag_config: RagConfig = RagConfig(),
        llm: RagLlm | None = None,
        checkpointer: Any | None = None,
        mcp_client: SupportMcpClient | None = None,
    ) -> SupportAgent:
        if mcp_client is None:
            from helpdeskai.mcp_servers.client import StdioMcpClient

            mcp_client = StdioMcpClient()
        graph = build_support_graph(
            intent_classifier=intent_classifier,
            config=config,
            rag_config=rag_config,
            llm=llm,
            checkpointer=checkpointer,
            mcp_client=mcp_client,
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
        return _run_graph_values(
            self.graph,
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

    def approve(self, *, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {"approval": "approved"})
        return _run_graph_values(self.graph, None, config=config)

    def reject(self, *, thread_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(config, {"approval": "rejected"})
        return _run_graph_values(self.graph, None, config=config)

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
