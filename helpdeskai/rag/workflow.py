from __future__ import annotations

import logging
import re
import json

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from helpdeskai.retrieval.config import RetrievalConfig
from helpdeskai.retrieval.io_utils import read_jsonl, write_json
from helpdeskai.retrieval.workflow import RetrievalEngine, SearchHit, tokenize

from .config import RagConfig
from .prompts import PROMPT_VARIANTS


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        """Return a retrieval-optimized query string."""


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage."""


class Generator(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_query: str,
        contexts: list[str],
    ) -> str:
        """Generate a grounded answer from prompt, query, and contexts."""


class Judge(Protocol):
    def evaluate(
        self,
        *,
        query: str,
        expected_answer: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Evaluate a generated answer on RAG quality metrics."""


class RetrievalLike(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Any | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        """Retrieve top matching chunks."""


@dataclass(slots=True)
class CompressedContext:
    chunk_id: str
    doc_id: str
    score: float
    source: str
    product: str
    compressed_text: str
    raw_text: str


class AnthropicMessagesClient:
    """Minimal Anthropic Messages API client with optional local mock mode."""

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
            # Deterministic response for local validation without network/API key.
            return user_prompt

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing")

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        request = Request(
            url=f"{self.api_base}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Anthropic API error {exc.code}: {error_body[:400]}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Anthropic API connection error: {exc}") from exc

        body = json.loads(raw_body)
        content = body.get("content", [])
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()


class HeuristicQueryRewriter:
    def rewrite(self, query: str) -> str:
        normalized = " ".join(query.split())
        sentences: list[str] = []
        seen: set[str] = set()
        for fragment in re.split(r"(?<=[?.!])\s+", normalized):
            cleaned = fragment.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            sentences.append(cleaned)
        return " ".join(sentences)


class LLMQueryRewriter:
    """Query rewriting through a dedicated prompt and Anthropic model."""

    REWRITE_PROMPT = (
        "Rewrite the user query for retrieval quality. "
        "Keep intent, product/version constraints, and error codes. "
        "Return only the rewritten query as one line."
    )

    def __init__(self, client: AnthropicMessagesClient, model: str) -> None:
        self._client = client
        self._model = model
        self._fallback = HeuristicQueryRewriter()

    def rewrite(self, query: str) -> str:
        if self._client.mock_mode:
            return self._fallback.rewrite(query)

        answer = self._client.complete(
            model=self._model,
            system_prompt=self.REWRITE_PROMPT,
            user_prompt=query,
            max_tokens=160,
            temperature=0.0,
        )
        cleaned = " ".join(answer.split())
        return cleaned if cleaned else self._fallback.rewrite(query)


class LexicalReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        scores: list[float] = []
        for passage in passages:
            passage_tokens = tokenize(passage)
            if not passage_tokens:
                scores.append(0.0)
                continue
            overlap = sum(1 for token in passage_tokens if token in query_tokens)
            density = overlap / len(passage_tokens)
            scores.append(overlap + density)
        return scores


class ExtractiveGenerator:
    def generate(
        self,
        *,
        system_prompt: str,
        user_query: str,
        contexts: list[str],
    ) -> str:
        if not contexts:
            return "I could not find enough grounded context to answer safely."
        ranked_sentences = _rank_sentences(user_query, contexts)
        if not ranked_sentences:
            return contexts[0].strip()
        if "avoid speculation" in system_prompt:
            return " ".join(sentence for sentence, _ in ranked_sentences[:2]).strip()
        if "short, actionable answer" in system_prompt:
            return ranked_sentences[0][0].strip()
        return " ".join(sentence for sentence, _ in ranked_sentences[:3]).strip()


class LLMGenerator:
    """Grounded answer generation through Anthropic, with deterministic mock fallback."""

    def __init__(self, client: AnthropicMessagesClient, model: str) -> None:
        self._client = client
        self._model = model
        self._fallback = ExtractiveGenerator()

    def generate(
        self,
        *,
        system_prompt: str,
        user_query: str,
        contexts: list[str],
    ) -> str:
        if self._client.mock_mode:
            return self._fallback.generate(
                system_prompt=system_prompt,
                user_query=user_query,
                contexts=contexts,
            )

        joined_context = "\n\n".join(f"[{idx + 1}] {ctx}" for idx, ctx in enumerate(contexts))
        user_prompt = (
            f"User query:\n{user_query}\n\n"
            f"Retrieved contexts:\n{joined_context}\n\n"
            "Answer using only the contexts. If not enough information is present, explicitly say so."
        )
        answer = self._client.complete(
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=600,
            temperature=0.1,
        )
        return answer.strip() if answer.strip() else self._fallback.generate(
            system_prompt=system_prompt,
            user_query=user_query,
            contexts=contexts,
        )


def _rank_sentences(query: str, contexts: list[str]) -> list[tuple[str, float]]:
    query_tokens = set(tokenize(query))
    ranked: list[tuple[str, float]] = []
    for context in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", context):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            sentence_tokens = tokenize(cleaned)
            if not sentence_tokens:
                continue
            overlap = sum(1 for token in sentence_tokens if token in query_tokens)
            density = overlap / len(sentence_tokens)
            ranked.append((cleaned, overlap + density))
    return sorted(ranked, key=lambda item: (-item[1], -len(item[0])))


def compress_context(query: str, text: str, *, max_sentences: int = 2) -> str:
    ranked_sentences = _rank_sentences(query, [text])
    if not ranked_sentences:
        return " ".join(text.split())
    selected = [sentence for sentence, _ in ranked_sentences[:max_sentences]]
    return " ".join(selected)


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize(" ".join(contexts)))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def answer_relevancy_score(query: str, answer: str) -> float:
    query_tokens = set(tokenize(query))
    answer_tokens = set(tokenize(answer))
    if not query_tokens or not answer_tokens:
        return 0.0
    return len(query_tokens & answer_tokens) / len(query_tokens)


def context_precision_score(query: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    query_tokens = set(tokenize(query))
    relevant_contexts = 0
    for context in contexts:
        if query_tokens & set(tokenize(context)):
            relevant_contexts += 1
    return relevant_contexts / len(contexts)


def context_recall_score(expected_answer: str, contexts: list[str]) -> float:
    expected_tokens = set(tokenize(expected_answer))
    if not expected_tokens:
        return 0.0
    context_tokens = set(tokenize(" ".join(contexts)))
    return len(expected_tokens & context_tokens) / len(expected_tokens)


class LLMJudge:
    """RAGAS-like metric scoring with an LLM judge."""

    def __init__(self, client: AnthropicMessagesClient, model: str) -> None:
        self._client = client
        self._model = model

    def evaluate(
        self,
        *,
        query: str,
        expected_answer: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        if self._client.mock_mode:
            return {
                "faithfulness": faithfulness_score(answer, contexts),
                "answer_relevancy": answer_relevancy_score(query, answer),
                "context_precision": context_precision_score(query, contexts),
                "context_recall": context_recall_score(expected_answer, contexts),
            }

        rubric = (
            "You are an LLM judge for RAG quality. Score each metric from 0.0 to 1.0.\n"
            "- faithfulness: answer grounded in contexts\n"
            "- answer_relevancy: answer addresses user query\n"
            "- context_precision: contexts are relevant to query\n"
            "- context_recall: contexts cover expected_answer\n"
            "Return ONLY JSON with keys: faithfulness, answer_relevancy, context_precision, context_recall"
        )
        prompt = json.dumps(
            {
                "query": query,
                "expected_answer": expected_answer,
                "answer": answer,
                "contexts": contexts,
            },
            ensure_ascii=True,
        )
        raw = self._client.complete(
            model=self._model,
            system_prompt=rubric,
            user_prompt=prompt,
            max_tokens=220,
            temperature=0.0,
        )
        metrics = _extract_metrics_from_text(raw)
        return {
            key: min(1.0, max(0.0, float(value)))
            for key, value in metrics.items()
        }


def _extract_metrics_from_text(text: str) -> dict[str, float]:
    json_match = re.search(r"\{[\s\S]*\}", text)
    payload = json.loads(json_match.group(0) if json_match else text)
    required = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Judge response missing keys: {sorted(missing)}")
    return {key: float(payload[key]) for key in required}


def load_baseline_report(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_faithfulness_regression(
    current_report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    max_drop: float,
) -> None:
    current_value = float(current_report["summary"]["best_faithfulness"])
    baseline_value = float(baseline_report["summary"]["best_faithfulness"])
    if baseline_value - current_value > max_drop:
        raise ValueError(
            "Faithfulness regression exceeds threshold: "
            f"baseline={baseline_value:.4f} current={current_value:.4f} max_drop={max_drop:.4f}"
        )


def build_retrieval_config(config: RagConfig) -> RetrievalConfig:
    return RetrievalConfig(
        chunks_path=config.chunks_path,
        golden_path=config.golden_path,
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
        collection_name=config.collection_name,
        embedding_model=config.embedding_model,
        mode=config.retrieval_mode,
        top_k=config.top_k,
    )


class RagEngine:
    def __init__(
        self,
        config: RagConfig,
        *,
        retrieval_engine: RetrievalLike | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.config = config
        self.config.validate(require_golden=False)
        self.llm_client = AnthropicMessagesClient(
            api_key=config.anthropic_api_key,
            api_base=config.anthropic_api_base,
            mock_mode=config.mock_llm,
        )
        self.retrieval_engine = retrieval_engine or RetrievalEngine(build_retrieval_config(config))
        self.query_rewriter = query_rewriter or LLMQueryRewriter(self.llm_client, config.rewrite_model)
        self.reranker = reranker or LexicalReranker()
        self.generator = generator or LLMGenerator(self.llm_client, config.generator_model)
        self.judge = judge or LLMJudge(self.llm_client, config.judge_model)

    def answer(
        self,
        query: str,
        *,
        prompt_variant: str | None = None,
        retrieval_mode: str | None = None,
    ) -> dict[str, Any]:
        active_prompt = prompt_variant or self.config.prompt_variant
        if active_prompt not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant: {active_prompt}")

        rewritten_query = self.query_rewriter.rewrite(query)
        step_trace: list[dict[str, Any]] = [
            {
                "step": "rewrite_query",
                "prompt": "rewrite_query_v1",
                "model": self.config.rewrite_model,
                "mock_llm": self.config.mock_llm,
                "input_query": query,
                "output_query": rewritten_query,
            }
        ]
        retrieved_hits = self.retrieval_engine.search(
            rewritten_query,
            top_k=self.config.top_k,
            mode=retrieval_mode or self.config.retrieval_mode,
        )
        step_trace.append(
            {
                "step": "retrieve",
                "mode": retrieval_mode or self.config.retrieval_mode,
                "requested_candidates": self.config.top_k,
                "returned_candidates": len(retrieved_hits),
            }
        )
        reranked_contexts = self._rerank_and_compress(rewritten_query, retrieved_hits)
        step_trace.append(
            {
                "step": "rerank",
                "model": self.config.reranker_model,
                "kept_candidates": len(reranked_contexts),
                "requested_keep": self.config.rerank_top_k,
            }
        )
        answer = self.generator.generate(
            system_prompt=PROMPT_VARIANTS[active_prompt],
            user_query=query,
            contexts=[context.compressed_text for context in reranked_contexts],
        )
        step_trace.append(
            {
                "step": "generate_answer",
                "model": self.config.generator_model,
                "mock_llm": self.config.mock_llm,
                "contexts_used": len(reranked_contexts),
            }
        )
        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "prompt_variant": active_prompt,
            "retrieval_mode": retrieval_mode or self.config.retrieval_mode,
            "answer": answer,
            "contexts": [asdict(context) for context in reranked_contexts],
            "generator_model": self.config.generator_model,
            "reranker_model": self.config.reranker_model,
            "steps": step_trace,
        }

    def _rerank_and_compress(self, query: str, hits: list[SearchHit]) -> list[CompressedContext]:
        if not hits:
            return []

        candidate_hits = hits[: max(self.config.top_k, self.config.rerank_top_k)]
        scores = self.reranker.score(query, [hit.text for hit in candidate_hits])
        scored_hits = sorted(
            zip(candidate_hits, scores, strict=True),
            key=lambda item: -item[1],
        )[: self.config.rerank_top_k]

        contexts: list[CompressedContext] = []
        for hit, score in scored_hits[: self.config.compression_top_k]:
            contexts.append(
                CompressedContext(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    score=float(score),
                    source=hit.source,
                    product=hit.product,
                    compressed_text=compress_context(query, hit.text),
                    raw_text=hit.text,
                )
            )
        return contexts

    def evaluate(
        self,
        *,
        sample_size: int | None = None,
    ) -> dict[str, Any]:
        self.config.validate(require_golden=True)
        rows = read_jsonl(self.config.golden_path)
        selected_rows = rows[: sample_size or self.config.evaluation_sample_size]
        variants: dict[str, Any] = {}

        for prompt_variant in PROMPT_VARIANTS:
            per_question: list[dict[str, Any]] = []
            totals = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

            for row in selected_rows:
                query = str(row.get("question", "")).strip()
                expected_answer = str(row.get("expected_answer", "")).strip()
                if not query:
                    continue

                payload = self.answer(query, prompt_variant=prompt_variant)
                contexts = [context["compressed_text"] for context in payload["contexts"]]
                metrics = self.judge.evaluate(
                    query=query,
                    expected_answer=expected_answer,
                    answer=payload["answer"],
                    contexts=contexts,
                )
                for key, value in metrics.items():
                    totals[key] += value

                per_question.append(
                    {
                        "question_id": row.get("question_id", ""),
                        "query": query,
                        "expected_answer": expected_answer,
                        "answer": payload["answer"],
                        "metrics": metrics,
                    }
                )

            count = len(per_question)
            variants[prompt_variant] = {
                "scores": {
                    key: (value / count if count else 0.0) for key, value in totals.items()
                },
                "questions": per_question,
            }

        best_prompt_variant = max(
            variants,
            key=lambda variant: (
                variants[variant]["scores"]["faithfulness"],
                variants[variant]["scores"]["answer_relevancy"],
            ),
        )
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "config": {
                "chunks_path": str(self.config.chunks_path),
                "golden_path": str(self.config.golden_path),
                "retrieval_mode": self.config.retrieval_mode,
                "rewrite_model": self.config.rewrite_model,
                "generator_model": self.config.generator_model,
                "judge_model": self.config.judge_model,
                "reranker_model": self.config.reranker_model,
                "mock_llm": self.config.mock_llm,
                "sample_size": len(selected_rows),
            },
            "prompt_variants": variants,
            "summary": {
                "best_prompt_variant": best_prompt_variant,
                "best_faithfulness": variants[best_prompt_variant]["scores"]["faithfulness"],
            },
        }


def run_rag_core(
    config: RagConfig,
    *,
    query: str,
    retrieval_engine: RetrievalLike | None = None,
    query_rewriter: QueryRewriter | None = None,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
) -> dict[str, Any]:
    logging.info("Starting RAG workflow with config: %s", asdict(config))
    engine = RagEngine(
        config,
        retrieval_engine=retrieval_engine,
        query_rewriter=query_rewriter,
        reranker=reranker,
        generator=generator,
        judge=judge,
    )
    return engine.answer(query)


def run_evaluation_core(
    config: RagConfig,
    *,
    sample_size: int | None = None,
    save_path: Path | None = None,
    baseline_path: Path | None = None,
    retrieval_engine: RetrievalLike | None = None,
    query_rewriter: QueryRewriter | None = None,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    judge: Judge | None = None,
) -> dict[str, Any]:
    logging.info("Starting RAG evaluation with config: %s", asdict(config))
    engine = RagEngine(
        config,
        retrieval_engine=retrieval_engine,
        query_rewriter=query_rewriter,
        reranker=reranker,
        generator=generator,
        judge=judge,
    )
    report = engine.evaluate(sample_size=sample_size)
    output_path = save_path or config.evaluation_path
    write_json(output_path, report)
    if baseline_path is not None:
        baseline_report = load_baseline_report(baseline_path)
        assert_faithfulness_regression(
            report,
            baseline_report,
            max_drop=config.max_faithfulness_drop,
        )
    return report