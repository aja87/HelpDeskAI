from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from helpdeskai.mcp_servers.crm import CRMService
from helpdeskai.mcp_servers.knoweldge import KnowledgeService


class MCPAgentState(TypedDict, total=False):
    query: str
    customer_id: str
    token: str
    top_k: int
    route: str
    subscription: dict[str, Any]
    answer: str
    contexts: list[dict[str, Any]]
    path_taken: list[str]


@dataclass(slots=True)
class MCPAgentConfig:
    token: str
    top_k: int = 5


class MCPOrchestrator:
    """LangGraph orchestration over CRM and Knowledge MCP-style services."""

    def __init__(
        self,
        config: MCPAgentConfig,
        *,
        crm_service: CRMService,
        knowledge_service: KnowledgeService,
    ) -> None:
        self.config = config
        self.crm_service = crm_service
        self.knowledge_service = knowledge_service

    @staticmethod
    def _is_billing_query(query: str) -> bool:
        lowered = query.lower()
        billing_tokens = {
            "billing",
            "invoice",
            "subscription",
            "payment",
            "renewal",
            "charged",
            "facturation",
            "abonnement",
        }
        return any(token in lowered for token in billing_tokens)

    def classify(self, state: MCPAgentState) -> dict[str, Any]:
        route = "billing" if self._is_billing_query(state["query"]) else "general"
        return {
            "route": route,
            "path_taken": list(state.get("path_taken", [])) + ["classify"],
        }

    def route(self, state: MCPAgentState) -> str:
        if state.get("route") == "billing":
            return "check_subscription"
        return "knowledge"

    def check_subscription(self, state: MCPAgentState) -> dict[str, Any]:
        customer_id = str(state.get("customer_id", "")).strip()
        if not customer_id:
            return {
                "answer": "I need a customer_id to handle billing-related questions.",
                "path_taken": list(state.get("path_taken", [])) + ["check_subscription"],
                "route": "blocked",
            }

        subscription = self.crm_service.get_subscription_status(
            customer_id=customer_id,
            token=state["token"],
        )
        return {
            "subscription": subscription,
            "path_taken": list(state.get("path_taken", [])) + ["check_subscription"],
            "route": "knowledge" if subscription.get("subscription_status") == "active" else "inactive_subscription",
        }

    def route_after_subscription(self, state: MCPAgentState) -> str:
        if state.get("route") == "knowledge":
            return "knowledge"
        return "inactive_subscription"

    def inactive_subscription(self, state: MCPAgentState) -> dict[str, Any]:
        subscription = state.get("subscription", {})
        status = subscription.get("subscription_status", "unknown")
        renewal_date = subscription.get("renewal_date", "unknown")
        answer = (
            "Before troubleshooting billing details, the subscription must be active. "
            f"Current status: {status}. Renewal date: {renewal_date}."
        )
        return {
            "answer": answer,
            "path_taken": list(state.get("path_taken", [])) + ["inactive_subscription"],
        }

    def knowledge(self, state: MCPAgentState) -> dict[str, Any]:
        payload = self.knowledge_service.answer_question(
            query=state["query"],
            token=state["token"],
            top_k=int(state.get("top_k", self.config.top_k)),
        )
        return {
            "answer": str(payload.get("answer", "")),
            "contexts": list(payload.get("contexts", [])),
            "path_taken": list(state.get("path_taken", [])) + ["knowledge"],
        }

    def build_graph(self):
        graph = StateGraph(MCPAgentState)
        graph.add_node("classify", self.classify)
        graph.add_node("check_subscription", self.check_subscription)
        graph.add_node("inactive_subscription", self.inactive_subscription)
        graph.add_node("knowledge", self.knowledge)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self.route,
            {
                "check_subscription": "check_subscription",
                "knowledge": "knowledge",
            },
        )
        graph.add_conditional_edges(
            "check_subscription",
            self.route_after_subscription,
            {
                "knowledge": "knowledge",
                "inactive_subscription": "inactive_subscription",
            },
        )
        graph.add_edge("knowledge", END)
        graph.add_edge("inactive_subscription", END)
        return graph.compile()


def run_mcp_agent_core(
    *,
    query: str,
    token: str,
    customer_id: str | None = None,
    top_k: int = 5,
    crm_service: CRMService,
    knowledge_service: KnowledgeService,
) -> dict[str, Any]:
    """Run the MCP-aware support orchestration graph for one request."""

    orchestrator = MCPOrchestrator(
        MCPAgentConfig(token=token, top_k=top_k),
        crm_service=crm_service,
        knowledge_service=knowledge_service,
    )
    graph = orchestrator.build_graph()
    state: MCPAgentState = {
        "query": query,
        "customer_id": customer_id or "",
        "token": token,
        "top_k": top_k,
        "path_taken": [],
    }
    result = graph.invoke(state)
    return {
        "query": query,
        "customer_id": customer_id or "",
        "answer": str(result.get("answer", "")),
        "subscription": result.get("subscription", {}),
        "contexts": list(result.get("contexts", [])),
        "path_taken": list(result.get("path_taken", [])),
        "route": str(result.get("route", "")),
    }
