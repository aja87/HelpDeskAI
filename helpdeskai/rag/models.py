"""Shared models for the advanced RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RagConfig:
    """Runtime configuration for Phase 4 RAG."""

    generator_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "claude-sonnet-5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_mode: str = "hybrid"
    prompt_version: str = "strict"
    candidate_k: int = 20
    final_k: int = 5
    max_context_chars: int = 8_000
    max_generation_tokens: int = 500
    max_rewrite_tokens: int = 120
    temperature: float = 0.0


@dataclass(frozen=True)
class StageTiming:
    """Latency measurement for one RAG stage."""

    name: str
    duration_ms: float


@dataclass(frozen=True)
class RagContext:
    """One context chunk selected for generation."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RagResult:
    """Structured result returned by the advanced RAG pipeline."""

    question_original: str
    question_rewritten: str
    answer: str
    contexts: list[RagContext]
    sources: list[str]
    timings: list[StageTiming]
    model_names: dict[str, str]
    prompt_version: str
    retrieval_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "question_original": self.question_original,
            "question_rewritten": self.question_rewritten,
            "answer": self.answer,
            "contexts": [asdict(context) for context in self.contexts],
            "sources": self.sources,
            "timings": [asdict(timing) for timing in self.timings],
            "model_names": dict(self.model_names),
            "prompt_version": self.prompt_version,
            "retrieval_mode": self.retrieval_mode,
        }
