from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import asyncpg
from langgraph.graph import END, START, StateGraph

from helpdeskai.rag.prompts import PROMPT_VARIANTS
from helpdeskai.rag.workflow import HeuristicQueryRewriter, LLMGenerator
from helpdeskai.retrieval.io_utils import read_jsonl
from helpdeskai.retrieval.workflow import SearchHit, SimpleBM25, tokenize

from .config import AgentsConfig


class RetrievalLike(Protocol):
	def search(
		self,
		query: str,
		*,
		top_k: int | None = None,
		filters: Any | None = None,
		mode: str = "hybrid",
	) -> list[SearchHit]:
		"""Return matching chunks for a query."""


class CheckpointStore(Protocol):
	def load(self, session_id: str) -> dict[str, Any] | None:
		"""Load the last saved state for a session."""

	def save(self, session_id: str, state: dict[str, Any]) -> None:
		"""Persist the current state for a session."""


class AgentState(TypedDict, total=False):
	session_id: str
	question: str
	history: list[dict[str, Any]]
	intent: str
	confidence: float
	clarification_needed: bool
	clarification: str
	route: str
	path_taken: list[str]
	rewritten_query: str
	retrieved_contexts: list[dict[str, Any]]
	answer: str
	rag_payload: dict[str, Any]
	token_estimate: int


class AnthropicMessagesClient:
	"""Minimal Anthropic Messages API client with deterministic mock fallback."""

	def __init__(
		self,
		*,
		api_key: str | None,
		api_base: str,
		mock_mode: bool,
		timeout_s: float = 45.0,
	) -> None:
		self.api_key = api_key
		self.api_base = api_base.rstrip("/")
		self.mock_mode = mock_mode
		self.timeout_s = timeout_s

	def complete(
		self,
		*,
		model: str,
		system_prompt: str,
		user_prompt: str,
		max_tokens: int,
		temperature: float = 0.0,
	) -> str:
		if self.mock_mode:
			return self._mock_complete(system_prompt=system_prompt, user_prompt=user_prompt)

		if not self.api_key:
			raise ValueError("ANTHROPIC_API_KEY is missing")

		payload = {
			"model": model,
			"max_tokens": max_tokens,
			"temperature": temperature,
			"system": system_prompt,
			"messages": [{"role": "user", "content": user_prompt}],
		}
		request = json.dumps(payload).encode("utf-8")
		from urllib.error import HTTPError, URLError
		from urllib.request import Request, urlopen

		headers = {
			"x-api-key": self.api_key,
			"anthropic-version": "2023-06-01",
			"content-type": "application/json",
		}
		http_request = Request(
			url=f"{self.api_base}/v1/messages",
			data=request,
			headers=headers,
			method="POST",
		)
		try:
			with urlopen(http_request, timeout=self.timeout_s) as response:  # noqa: S310
				raw_body = response.read().decode("utf-8")
		except HTTPError as exc:
			error_body = exc.read().decode("utf-8", errors="replace")
			raise RuntimeError(f"Anthropic API error {exc.code}: {error_body[:400]}") from exc
		except URLError as exc:
			raise RuntimeError(f"Anthropic API connection error: {exc}") from exc

		body = json.loads(raw_body)
		content = body.get("content", [])
		parts: list[str] = []
		for block in content:
			if isinstance(block, dict) and block.get("type") == "text":
				parts.append(str(block.get("text", "")))
		return "\n".join(parts).strip()

	def _mock_complete(self, *, system_prompt: str, user_prompt: str) -> str:
		lowered_system = system_prompt.lower()
		lowered_user = user_prompt.lower()

		if "json" in lowered_system and ("intent" in lowered_system or "classifier" in lowered_system):
			intent = _heuristic_intent(user_prompt)
			confidence = 0.92 if intent in {"chitchat", "escalate"} else 0.78
			if intent == "clarify":
				confidence = 0.54
			clarification = _heuristic_clarification(user_prompt)
			return json.dumps(
				{
					"intent": intent,
					"confidence": confidence,
					"clarification": clarification,
					"reason": "mock classifier",
				},
				ensure_ascii=True,
			)

		if "clarifying question" in lowered_system:
			return _heuristic_clarification(user_prompt)

		if "brief, friendly" in lowered_system or "conversationnel" in lowered_system:
			return _mock_chitchat_response(lowered_user)

		if "support assistant" in lowered_system or "grounded" in lowered_system:
			return _mock_grounded_response(user_prompt)

		return user_prompt


class LocalChunkRetrievalEngine:
	"""Deterministic fallback retriever over processed JSONL chunks."""

	def __init__(self, chunks_path: Path) -> None:
		self._chunks = read_jsonl(chunks_path)
		if not self._chunks:
			raise ValueError(f"No chunks found in {chunks_path}")
		self._chunk_ids = [str(row.get("chunk_id", "")) for row in self._chunks]
		self._bm25 = SimpleBM25([tokenize(str(row.get("text", ""))) for row in self._chunks])

	def search(
		self,
		query: str,
		*,
		top_k: int | None = None,
		filters: Any | None = None,
		mode: str = "hybrid",
	) -> list[SearchHit]:
		del filters, mode
		limit = top_k or 5
		scores = self._bm25.get_scores(tokenize(query))
		ranked_indices = sorted(range(len(scores)), key=lambda index: -scores[index])
		hits: list[SearchHit] = []
		for index in ranked_indices[:limit]:
			row = self._chunks[index]
			hits.append(
				SearchHit(
					chunk_id=str(row.get("chunk_id", "")),
					doc_id=str(row.get("doc_id", "")),
					score=float(scores[index]),
					text=str(row.get("text", "")),
					source=str(row.get("source", "")),
					product=str(row.get("product", "")),
					version=str(row.get("version", "")),
					category=str(row.get("category", "")),
					date=str(row.get("date", "")),
				)
			)
		return hits


@dataclass(slots=True)
class SQLiteCheckpointStore:
	path: Path

	def __post_init__(self) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		with sqlite3.connect(self.path) as connection:
			connection.execute(
				"""
				CREATE TABLE IF NOT EXISTS agent_checkpoints (
					session_id TEXT PRIMARY KEY,
					state_json TEXT NOT NULL,
					updated_at TEXT NOT NULL
				)
				"""
			)
			connection.commit()

	def load(self, session_id: str) -> dict[str, Any] | None:
		with sqlite3.connect(self.path) as connection:
			row = connection.execute(
				"SELECT state_json FROM agent_checkpoints WHERE session_id = ?",
				(session_id,),
			).fetchone()
		if row is None:
			return None
		return json.loads(row[0])

	def save(self, session_id: str, state: dict[str, Any]) -> None:
		payload = json.dumps(state, ensure_ascii=True)
		with sqlite3.connect(self.path) as connection:
			connection.execute(
				"""
				INSERT INTO agent_checkpoints(session_id, state_json, updated_at)
				VALUES(?, ?, ?)
				ON CONFLICT(session_id) DO UPDATE SET
					state_json = excluded.state_json,
					updated_at = excluded.updated_at
				""",
				(session_id, payload, datetime.now(UTC).isoformat()),
			)
			connection.commit()


@dataclass(slots=True)
class PostgresCheckpointStore:
	dsn: str

	async def _init(self) -> None:
		connection = await asyncpg.connect(self.dsn)
		try:
			await connection.execute(
				"""
				CREATE TABLE IF NOT EXISTS agent_checkpoints (
					session_id TEXT PRIMARY KEY,
					state_json TEXT NOT NULL,
					updated_at TIMESTAMPTZ NOT NULL
				)
				"""
			)
		finally:
			await connection.close()

	async def _load(self, session_id: str) -> dict[str, Any] | None:
		connection = await asyncpg.connect(self.dsn)
		try:
			row = await connection.fetchrow(
				"SELECT state_json FROM agent_checkpoints WHERE session_id = $1",
				session_id,
			)
		finally:
			await connection.close()
		if row is None:
			return None
		return json.loads(row[0])

	async def _save(self, session_id: str, state: dict[str, Any]) -> None:
		payload = json.dumps(state, ensure_ascii=True)
		connection = await asyncpg.connect(self.dsn)
		try:
			await connection.execute(
				"""
				INSERT INTO agent_checkpoints(session_id, state_json, updated_at)
				VALUES($1, $2, $3)
				ON CONFLICT (session_id) DO UPDATE SET
					state_json = EXCLUDED.state_json,
					updated_at = EXCLUDED.updated_at
				""",
				session_id,
				payload,
				datetime.now(UTC),
			)
		finally:
			await connection.close()

	def load(self, session_id: str) -> dict[str, Any] | None:
		return asyncio.run(self._load(session_id))

	def save(self, session_id: str, state: dict[str, Any]) -> None:
		asyncio.run(self._save(session_id, state))


def _load_env_file(path: Path) -> None:
	if not path.exists():
		return
	for line in path.read_text(encoding="utf-8").splitlines():
		stripped = line.strip()
		if not stripped or stripped.startswith("#") or "=" not in stripped:
			continue
		key, value = stripped.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


def _heuristic_intent(question: str) -> str:
	question_text = _extract_question_text(question)
	lowered = question_text.lower()
	if any(token in lowered for token in ["delete", "remove", "destroy", "secret", "token", "credential", "refund", "billing", "chargeback", "admin", "security"]):
		return "escalate"
	if any(token in lowered for token in ["hello", "hi", "hey", "thanks", "thank you", "how are you", "who are you", "joke"]):
		return "chitchat"
	if len(question_text.split()) < 4 or question_text.strip().endswith("?") and len(question_text.split()) < 5:
		return "clarify"
	return "factual"


def _heuristic_clarification(question: str) -> str:
	lowered = _extract_question_text(question).lower()
	if "billing" in lowered or "refund" in lowered:
		return "Can you clarify the account, invoice, or subscription issue you want help with?"
	if "error" in lowered or "problem" in lowered:
		return "Can you share the exact error message and the product or version involved?"
	return "Can you add the missing context or the specific outcome you want?"


def _mock_chitchat_response(question: str) -> str:
	if any(token in question for token in ["hi", "hello", "hey"]):
		return "Hello. I can help with support questions, troubleshooting, or account issues."
	return "I’m here if you want help with a support or troubleshooting question."


def _extract_question_text(prompt: str) -> str:
	if "Question:" in prompt:
		return prompt.split("Question:", 1)[1].strip()
	return prompt.strip()


def _mock_grounded_response(user_prompt: str) -> str:
	match = re.search(r"Retrieved contexts:\n([\s\S]*?)\n\nAnswer", user_prompt)
	if match:
		contexts = [line.strip(" []") for line in match.group(1).splitlines() if line.strip()]
		if contexts:
			return contexts[0]
	return "I could not find enough grounded context to answer safely."


def _ensure_list(value: Any) -> list[Any]:
	if isinstance(value, list):
		return value
	return []


class SupportAgentRuntime:
	def __init__(
		self,
		config: AgentsConfig,
		*,
		retrieval_engine: RetrievalLike | None = None,
		checkpoint_store: CheckpointStore | None = None,
	) -> None:
		config.validate()
		self.config = config
		self.llm_client = AnthropicMessagesClient(
			api_key=config.anthropic_api_key,
			api_base=config.anthropic_api_base,
			mock_mode=config.mock_llm,
		)
		self.query_rewriter = HeuristicQueryRewriter()
		self.generator = LLMGenerator(self.llm_client, config.generator_model)
		self.retrieval_engine = retrieval_engine or self._build_retrieval_engine()
		self.checkpoint_store = checkpoint_store or self._build_checkpoint_store()

	def _build_retrieval_engine(self) -> RetrievalLike:
		try:
			from helpdeskai.retrieval.config import RetrievalConfig
			from helpdeskai.retrieval.workflow import RetrievalEngine

			retrieval_config = RetrievalConfig(
				chunks_path=self.config.chunks_path,
				qdrant_url=self.config.qdrant_url,
				qdrant_api_key=self.config.qdrant_api_key,
				collection_name=self.config.collection_name,
				embedding_model=self.config.embedding_model,
			)
			return RetrievalEngine(retrieval_config)
		except Exception as exc:  # pragma: no cover - fallback path is exercised in tests
			logging.warning("Falling back to local chunk retrieval: %s", exc)
			return LocalChunkRetrievalEngine(self.config.chunks_path)

	def _build_checkpoint_store(self) -> CheckpointStore:
		if self.config.checkpoint_backend == "postgres":
			return PostgresCheckpointStore(self.config.checkpoint_dsn or "")
		return SQLiteCheckpointStore(self.config.checkpoint_path)

	def build_initial_state(self, question: str, session_id: str) -> AgentState:
		prior_state = self.checkpoint_store.load(session_id) or {}
		history = _ensure_list(prior_state.get("history"))
		history.append(
			{
				"role": "user",
				"content": question,
				"at": datetime.now(UTC).isoformat(),
			}
		)
		return {
			"session_id": session_id,
			"question": question,
			"history": history,
			"path_taken": [],
		}

	def classify_intent(self, state: AgentState) -> dict[str, Any]:
		prompt = (
			"Classify the support request into exactly one category: factual, chitchat, "
			"escalate, or clarify. Return JSON only with keys intent, confidence, clarification. "
			"Use escalate for sensitive actions, credential handling, account deletion, billing changes, "
			"or any security-risk request. Use clarify when the request is underspecified.\n\n"
			f"Question: {state['question']}"
		)
		raw = self.llm_client.complete(
			model=self.config.classifier_model,
			system_prompt="You are a support intent classifier. Return JSON only with intent, confidence, clarification.",
			user_prompt=prompt,
			max_tokens=120,
			temperature=0.0,
		)
		decision = _parse_intent(raw)
		clarification_needed = decision["intent"] == "clarify" or decision["confidence"] < self.config.confidence_threshold
		route = "clarify" if clarification_needed else decision["intent"]
		path_taken = list(state.get("path_taken", [])) + ["classify_intent"]
		return {
			"intent": decision["intent"],
			"confidence": decision["confidence"],
			"clarification_needed": clarification_needed,
			"clarification": decision.get("clarification", ""),
			"route": route,
			"path_taken": path_taken,
		}

	def route_by_intent(self, state: AgentState) -> Literal["retrieve", "chitchat", "clarify", "escalate"]:
		if state.get("clarification_needed"):
			return "clarify"
		intent = state.get("intent", "clarify")
		if intent == "factual":
			return "retrieve"
		if intent == "chitchat":
			return "chitchat"
		if intent == "escalate":
			return "escalate"
		return "clarify"

	def retrieve(self, state: AgentState) -> dict[str, Any]:
		rewritten_query = self.query_rewriter.rewrite(state["question"])
		hits = self.retrieval_engine.search(
			rewritten_query,
			top_k=self.config.top_k,
			mode="hybrid",
		)
		ranked_hits = _rank_hits(rewritten_query, hits, top_k=self.config.top_k)
		contexts = [_compress_hit(rewritten_query, hit) for hit in ranked_hits]
		path_taken = list(state.get("path_taken", [])) + ["retrieve"]
		return {
			"rewritten_query": rewritten_query,
			"retrieved_contexts": contexts,
			"path_taken": path_taken,
			"token_estimate": _estimate_tokens(state.get("question", ""), contexts),
		}

	def generate(self, state: AgentState) -> dict[str, Any]:
		contexts = state.get("retrieved_contexts", [])
		context_texts = [str(context.get("compressed_text", "")) for context in contexts if context]
		answer = self.generator.generate(
			system_prompt=PROMPT_VARIANTS["grounded"],
			user_query=state["question"],
			contexts=context_texts,
		)
		path_taken = list(state.get("path_taken", [])) + ["generate"]
		return {
			"answer": answer,
			"path_taken": path_taken,
			"rag_payload": {
				"query": state["question"],
				"rewritten_query": state.get("rewritten_query", state["question"]),
				"contexts": contexts,
				"answer": answer,
			},
		}

	def chitchat(self, state: AgentState) -> dict[str, Any]:
		answer = self.llm_client.complete(
			model=self.config.generator_model,
			system_prompt="You are a brief, friendly support assistant for light conversation.",
			user_prompt=state["question"],
			max_tokens=120,
			temperature=0.3,
		)
		path_taken = list(state.get("path_taken", [])) + ["chitchat"]
		return {"answer": answer, "path_taken": path_taken}

	def clarify(self, state: AgentState) -> dict[str, Any]:
		clarification = state.get("clarification") or self.llm_client.complete(
			model=self.config.classifier_model,
			system_prompt="Return one concise clarification question for a support request.",
			user_prompt=state["question"],
			max_tokens=80,
			temperature=0.0,
		)
		path_taken = list(state.get("path_taken", [])) + ["clarify"]
		return {
			"clarification": clarification,
			"answer": clarification,
			"path_taken": path_taken,
		}

	def escalate(self, state: AgentState) -> dict[str, Any]:
		answer = (
			"This request requires human review for security or compliance reasons. "
			"I have escalated it to the support team."
		)
		path_taken = list(state.get("path_taken", [])) + ["escalate"]
		return {"answer": answer, "path_taken": path_taken}

	def build_graph(self):
		graph = StateGraph(AgentState)
		graph.add_node("classify_intent", self.classify_intent)
		graph.add_node("retrieve", self.retrieve)
		graph.add_node("generate", self.generate)
		graph.add_node("chitchat", self.chitchat)
		graph.add_node("clarify", self.clarify)
		graph.add_node("escalate", self.escalate)

		graph.add_edge(START, "classify_intent")
		graph.add_conditional_edges(
			"classify_intent",
			self.route_by_intent,
			{
				"retrieve": "retrieve",
				"chitchat": "chitchat",
				"clarify": "clarify",
				"escalate": "escalate",
			},
		)
		graph.add_edge("retrieve", "generate")
		graph.add_edge("generate", END)
		graph.add_edge("chitchat", END)
		graph.add_edge("clarify", END)
		graph.add_edge("escalate", END)
		return graph.compile()


def _parse_intent(raw: str) -> dict[str, Any]:
	match = re.search(r"\{[\s\S]*\}", raw)
	payload = json.loads(match.group(0) if match else raw)
	intent = str(payload.get("intent", "clarify")).strip().lower()
	if intent not in {"factual", "chitchat", "escalate", "clarify"}:
		intent = _heuristic_intent(str(payload.get("clarification", "")) or raw)
	confidence = float(payload.get("confidence", 0.0))
	if confidence < 0.0:
		confidence = 0.0
	if confidence > 1.0:
		confidence = 1.0
	clarification = str(payload.get("clarification", "")).strip()
	if not clarification and intent == "clarify":
		clarification = _heuristic_clarification(raw)
	return {"intent": intent, "confidence": confidence, "clarification": clarification}


def _rank_hits(query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
	if not hits:
		return []
	query_tokens = set(tokenize(query))
	scored_hits: list[tuple[SearchHit, float]] = []
	for hit in hits:
		hit_tokens = tokenize(hit.text)
		overlap = sum(1 for token in hit_tokens if token in query_tokens)
		density = overlap / len(hit_tokens) if hit_tokens else 0.0
		scored_hits.append((hit, overlap + density + hit.score))
	return [hit for hit, _ in sorted(scored_hits, key=lambda item: -item[1])[:top_k]]


def _compress_hit(query: str, hit: SearchHit) -> dict[str, Any]:
	compressed_text = _compress_text(query, hit.text)
	return {
		"chunk_id": hit.chunk_id,
		"doc_id": hit.doc_id,
		"score": hit.score,
		"source": hit.source,
		"product": hit.product,
		"version": hit.version,
		"category": hit.category,
		"date": hit.date,
		"compressed_text": compressed_text,
		"raw_text": hit.text,
	}


def _compress_text(query: str, text: str, *, max_sentences: int = 2) -> str:
	query_tokens = set(tokenize(query))
	candidates: list[tuple[str, float]] = []
	for fragment in re.split(r"(?<=[.!?])\s+|\n+", text):
		cleaned = fragment.strip()
		if not cleaned:
			continue
		hit_tokens = tokenize(cleaned)
		if not hit_tokens:
			continue
		overlap = sum(1 for token in hit_tokens if token in query_tokens)
		density = overlap / len(hit_tokens)
		candidates.append((cleaned, overlap + density))
	if not candidates:
		return " ".join(text.split())
	selected = [sentence for sentence, _ in sorted(candidates, key=lambda item: -item[1])[:max_sentences]]
	return " ".join(selected)


def _estimate_tokens(question: str, contexts: list[dict[str, Any]]) -> int:
	total_text = question + " " + " ".join(str(context.get("compressed_text", "")) for context in contexts)
	return len(total_text.split())


def _build_checkpoint_store(config: AgentsConfig) -> CheckpointStore:
	if config.checkpoint_backend == "postgres":
		return PostgresCheckpointStore(config.checkpoint_dsn or "")
	return SQLiteCheckpointStore(config.checkpoint_path)


def _build_runtime(
	config: AgentsConfig,
	*,
	retrieval_engine: RetrievalLike | None = None,
	checkpoint_store: CheckpointStore | None = None,
) -> SupportAgentRuntime:
	return SupportAgentRuntime(
		config,
		retrieval_engine=retrieval_engine,
		checkpoint_store=checkpoint_store or _build_checkpoint_store(config),
	)


def run_agents_core(
	config: AgentsConfig,
	*,
	query: str,
	session_id: str | None = None,
	retrieval_engine: RetrievalLike | None = None,
	checkpoint_store: CheckpointStore | None = None,
) -> dict[str, Any]:
	"""Run the LangGraph agent workflow for a single user query."""

	active_session_id = session_id or config.session_id
	logging.info("Starting agents workflow with config: %s", asdict(config))
	runtime = _build_runtime(
		config,
		retrieval_engine=retrieval_engine,
		checkpoint_store=checkpoint_store,
	)
	graph = runtime.build_graph()
	state = runtime.build_initial_state(query, active_session_id)
	result = graph.invoke(state)

	history = _ensure_list(result.get("history"))
	history.append(
		{
			"role": "assistant",
			"content": result.get("answer", ""),
			"at": datetime.now(UTC).isoformat(),
		}
	)
	final_state = {
		"session_id": active_session_id,
		"question": query,
		"intent": result.get("intent", "clarify"),
		"confidence": float(result.get("confidence", 0.0)),
		"clarification": result.get("clarification", ""),
		"answer": result.get("answer", ""),
		"path_taken": _ensure_list(result.get("path_taken")),
		"history": history,
		"rewritten_query": result.get("rewritten_query", query),
		"retrieved_contexts": _ensure_list(result.get("retrieved_contexts")),
		"rag_payload": result.get("rag_payload", {}),
		"token_estimate": int(result.get("token_estimate", 0)),
	}
	runtime.checkpoint_store.save(active_session_id, final_state)
	return final_state


def build_graph(
	config: AgentsConfig,
	*,
	retrieval_engine: RetrievalLike | None = None,
	checkpoint_store: CheckpointStore | None = None,
):
	"""Build the compiled LangGraph agent for reuse in scripts or tests."""

	runtime = _build_runtime(
		config,
		retrieval_engine=retrieval_engine,
		checkpoint_store=checkpoint_store,
	)
	return runtime.build_graph()


def export_graph_core(
	config: AgentsConfig,
	*,
	output_path: Path | None = None,
) -> dict[str, Any]:
	"""Export the workflow graph as Mermaid text."""

	runtime = _build_runtime(config)
	graph = runtime.build_graph()
	target_path = output_path or config.graph_path
	target_path.parent.mkdir(parents=True, exist_ok=True)
	mermaid = graph.get_graph().draw_mermaid()
	target_path.write_text(mermaid, encoding="utf-8")
	return {"graph_path": str(target_path), "format": "mermaid"}

