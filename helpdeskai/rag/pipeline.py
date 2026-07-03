"""Advanced RAG pipeline: rewrite, retrieve, rerank, compress, generate."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from helpdeskai.rag.llm import ClaudeLlm, RagLlm
from helpdeskai.rag.models import RagConfig, RagContext, RagResult, StageTiming
from helpdeskai.rag.prompts import REWRITE_PROMPT, get_prompt_variant
from helpdeskai.rag.rerank import CrossEncoderReranker, Reranker
from helpdeskai.retrieval.models import SearchResult
from helpdeskai.retrieval.search import SearchEngine


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def compress_contexts(
    contexts: Sequence[SearchResult],
    *,
    max_chars: int,
) -> list[RagContext]:
    """Pack selected contexts under a character budget."""
    compressed: list[RagContext] = []
    used = 0
    for context in contexts:
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = context.content.strip()
        if len(content) > remaining:
            content = content[: max(0, remaining)].rstrip()
        if not content:
            continue
        compressed.append(
            RagContext(
                chunk_id=context.chunk_id,
                document_id=context.document_id,
                content=content,
                score=context.score,
                metadata=dict(context.metadata),
                source_scores=dict(context.source_scores),
            )
        )
        used += len(content)
    return compressed


def format_context(contexts: Sequence[RagContext]) -> str:
    """Format contexts with chunk IDs for source-aware generation."""
    return "\n\n".join(f"[{context.chunk_id}] {context.content}" for context in contexts)


class AdvancedRagPipeline:
    """Phase 4 RAG pipeline composed from injectable stages."""

    def __init__(
        self,
        *,
        config: RagConfig = RagConfig(),
        llm: RagLlm | None = None,
        search_engine: SearchEngine | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.search_engine = search_engine
        self.reranker = reranker

    def _llm(self) -> RagLlm:
        if self.llm is None:
            self.llm = ClaudeLlm()
        return self.llm

    def _search_engine(self) -> SearchEngine:
        if self.search_engine is None:
            self.search_engine = SearchEngine()
        return self.search_engine

    def _reranker(self) -> Reranker:
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(self.config.reranker_model)
        return self.reranker

    def rewrite_query(self, question: str) -> str:
        """Rewrite a user question into a retrieval-oriented query."""
        rewritten = self._llm().complete(
            REWRITE_PROMPT.format(question=question),
            model=self.config.generator_model,
            max_tokens=self.config.max_rewrite_tokens,
            temperature=self.config.temperature,
        )
        return rewritten.strip() or question

    def retrieve(self, rewritten_question: str) -> list[SearchResult]:
        """Retrieve candidates through the existing retrieval layer."""
        return self._search_engine().search(
            rewritten_question,
            top_k=self.config.candidate_k,
            mode=self.config.retrieval_mode,
        )

    def generate(self, question: str, contexts: Sequence[RagContext]) -> str:
        """Generate a grounded answer from selected contexts."""
        prompt = get_prompt_variant(self.config.prompt_version)(
            question,
            format_context(contexts),
        )
        return self._llm().complete(
            prompt,
            model=self.config.generator_model,
            max_tokens=self.config.max_generation_tokens,
            temperature=self.config.temperature,
        )

    def run(self, question: str) -> RagResult:
        """Execute rewrite -> retrieval -> rerank -> compression -> generation."""
        timings: list[StageTiming] = []

        started = time.perf_counter()
        rewritten = self.rewrite_query(question)
        timings.append(StageTiming("1. Query rewriting", _elapsed_ms(started)))

        started = time.perf_counter()
        candidates = self.retrieve(rewritten)
        timings.append(StageTiming("2. Retrieval", _elapsed_ms(started)))

        started = time.perf_counter()
        reranked = self._reranker().rerank(rewritten, candidates, top_k=self.config.final_k)
        timings.append(StageTiming("3. Re-ranking", _elapsed_ms(started)))

        started = time.perf_counter()
        contexts = compress_contexts(reranked, max_chars=self.config.max_context_chars)
        timings.append(StageTiming("4. Context compression", _elapsed_ms(started)))

        started = time.perf_counter()
        answer = self.generate(question, contexts)
        timings.append(StageTiming("5. Generation", _elapsed_ms(started)))

        return RagResult(
            question_original=question,
            question_rewritten=rewritten,
            answer=answer,
            contexts=contexts,
            sources=[context.chunk_id for context in contexts],
            timings=timings,
            model_names={
                "generator": self.config.generator_model,
                "reranker": self._reranker().model_name,
            },
            prompt_version=self.config.prompt_version,
            retrieval_mode=self.config.retrieval_mode,
        )

    def run_many(self, questions: Sequence[str]) -> list[RagResult]:
        """Run the pipeline over multiple questions."""
        return [self.run(question) for question in questions]


def result_to_ragas_row(result: RagResult, ground_truth: str | None = None) -> dict[str, Any]:
    """Convert a RAG result to a Ragas-compatible row."""
    row: dict[str, Any] = {
        "question": result.question_original,
        "answer": result.answer,
        "contexts": [context.content for context in result.contexts],
    }
    if ground_truth is not None:
        row["ground_truth"] = ground_truth
    return row
