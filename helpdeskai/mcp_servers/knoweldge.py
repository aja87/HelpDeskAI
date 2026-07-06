from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from helpdeskai.rag.workflow import compress_context
from helpdeskai.retrieval.io_utils import read_jsonl
from helpdeskai.retrieval.workflow import SimpleBM25, tokenize


class MCPAuthError(PermissionError):
	"""Raised when a tool call is not authenticated."""


class MCPRateLimitError(RuntimeError):
	"""Raised when a token exceeds the local rate limit."""


class ToolInputError(ValueError):
	"""Raised when tool input validation fails."""


class _StrictModel(BaseModel):
	model_config = ConfigDict(extra="forbid", strict=True)


class SearchKnowledgeInput(_StrictModel):
	query: str = Field(min_length=3, max_length=2000)
	token: str = Field(min_length=8, max_length=256)
	top_k: int = Field(default=5, ge=1, le=20)
	product: str | None = Field(default=None, max_length=64)
	version: str | None = Field(default=None, max_length=64)
	category: str | None = Field(default=None, max_length=64)


class AnswerQuestionInput(_StrictModel):
	query: str = Field(min_length=3, max_length=2000)
	token: str = Field(min_length=8, max_length=256)
	top_k: int = Field(default=5, ge=1, le=20)


@dataclass(slots=True)
class RateLimiter:
	max_calls: int
	period_seconds: int
	_windows: dict[str, deque[float]] = field(init=False, repr=False)

	def __post_init__(self) -> None:
		self._windows = defaultdict(deque)

	def consume(self, key: str) -> None:
		now = time.monotonic()
		window = self._windows[key]
		threshold = now - float(self.period_seconds)
		while window and window[0] < threshold:
			window.popleft()
		if len(window) >= self.max_calls:
			raise MCPRateLimitError(
				f"Rate limit exceeded for key={key}. max_calls={self.max_calls}, period_seconds={self.period_seconds}"
			)
		window.append(now)


class KnowledgeService:
	"""Knowledge search service exposed through MCP tools."""

	TOOL_DOCS = {
		"search_knowledge": {
			"summary": "Search technical knowledge chunks with optional metadata filters.",
			"inputs": {
				"query": "str",
				"token": "str",
				"top_k": "int=5",
				"product": "str|None",
				"version": "str|None",
				"category": "str|None",
			},
			"returns": "list[dict] ranked chunks",
		},
		"answer_question": {
			"summary": "Build a grounded answer from top retrieved knowledge contexts.",
			"inputs": {"query": "str", "token": "str", "top_k": "int=5"},
			"returns": "dict with answer and supporting contexts",
		},
	}

	def __init__(
		self,
		*,
		chunks_path: Path = Path("data/processed/techqa_chunks.jsonl"),
		expected_token: str | None = None,
		rate_limit_calls: int = 80,
		rate_limit_window_s: int = 60,
	) -> None:
		self.expected_token = expected_token or os.getenv("HELPDESKAI_MCP_TOKEN", "dev-token-unsafe")
		self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, period_seconds=rate_limit_window_s)
		self.chunks = read_jsonl(chunks_path)
		if not self.chunks:
			raise ValueError(f"No chunks found in {chunks_path}")
		self._bm25 = SimpleBM25([tokenize(str(row.get("text", ""))) for row in self.chunks])

	@staticmethod
	def describe_tools() -> dict[str, Any]:
		return KnowledgeService.TOOL_DOCS

	def _authorize(self, token: str) -> None:
		if token != self.expected_token:
			raise MCPAuthError("Invalid MCP token")
		self.rate_limiter.consume(token)

	@staticmethod
	def _validate(model: type[_StrictModel], payload: dict[str, Any]) -> _StrictModel:
		try:
			return model.model_validate(payload)
		except ValidationError as exc:
			raise ToolInputError(str(exc)) from exc

	@staticmethod
	def _matches_filters(
		row: dict[str, Any],
		*,
		product: str | None,
		version: str | None,
		category: str | None,
	) -> bool:
		if product and row.get("product") != product:
			return False
		if version and row.get("version") != version:
			return False
		if category and row.get("category") != category:
			return False
		return True

	def search_knowledge(
		self,
		*,
		query: str,
		token: str,
		top_k: int = 5,
		product: str | None = None,
		version: str | None = None,
		category: str | None = None,
	) -> list[dict[str, Any]]:
		req = self._validate(
			SearchKnowledgeInput,
			{
				"query": query,
				"token": token,
				"top_k": top_k,
				"product": product,
				"version": version,
				"category": category,
			},
		)
		self._authorize(req.token)

		scores = self._bm25.get_scores(tokenize(req.query))
		ranked_indices = sorted(range(len(scores)), key=lambda index: -scores[index])
		rows: list[dict[str, Any]] = []
		for index in ranked_indices:
			row = self.chunks[index]
			if not self._matches_filters(
				row,
				product=req.product,
				version=req.version,
				category=req.category,
			):
				continue
			rows.append(
				{
					"chunk_id": str(row.get("chunk_id", "")),
					"doc_id": str(row.get("doc_id", "")),
					"score": float(scores[index]),
					"product": str(row.get("product", "")),
					"version": str(row.get("version", "")),
					"category": str(row.get("category", "")),
					"source": str(row.get("source", "")),
					"compressed_text": compress_context(req.query, str(row.get("text", ""))),
					"text": str(row.get("text", "")),
				}
			)
			if len(rows) >= req.top_k:
				break
		return rows

	def answer_question(self, *, query: str, token: str, top_k: int = 5) -> dict[str, Any]:
		req = self._validate(
			AnswerQuestionInput,
			{"query": query, "token": token, "top_k": top_k},
		)
		self._authorize(req.token)
		contexts = self.search_knowledge(query=req.query, token=req.token, top_k=req.top_k)
		if not contexts:
			return {
				"answer": "I could not find enough grounded context to answer safely.",
				"contexts": [],
			}
		answer = " ".join(row["compressed_text"] for row in contexts[:2]).strip()
		return {"answer": answer, "contexts": contexts}


def _build_mcp_app(service: KnowledgeService):
	try:
		from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]
	except Exception as exc:  # pragma: no cover
		logging.exception("Failed to import FastMCP for Knowledge server")
		raise RuntimeError("mcp package is required to run the Knowledge MCP server") from exc

	mcp = FastMCP("knowledge-server")

	@mcp.tool()
	def search_knowledge(
		query: str,
		token: str,
		top_k: int = 5,
		product: str | None = None,
		version: str | None = None,
		category: str | None = None,
	) -> list[dict[str, Any]]:
		"""Search technical KB chunks and return ranked contexts."""

		return service.search_knowledge(
			query=query,
			token=token,
			top_k=top_k,
			product=product,
			version=version,
			category=category,
		)

	@mcp.tool()
	def answer_question(query: str, token: str, top_k: int = 5) -> dict[str, Any]:
		"""Answer with grounded content extracted from retrieved contexts."""

		return service.answer_question(query=query, token=token, top_k=top_k)

	return mcp


def run_server() -> None:
	logging.info("Initializing Knowledge service")
	service = KnowledgeService()
	logging.info("Knowledge service ready: chunks=%d", len(service.chunks))
	app = _build_mcp_app(service)
	logging.info("Knowledge MCP server bootstrapped; entering FastMCP run loop")
	app.run()


if __name__ == "__main__":
	run_server()
