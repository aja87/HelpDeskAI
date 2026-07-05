from __future__ import annotations

import json

from pathlib import Path

from helpdeskai.rag.config import RagConfig
from helpdeskai.rag.prompts import PROMPT_VARIANTS
from helpdeskai.rag.workflow import (
    ExtractiveGenerator,
    HeuristicQueryRewriter,
    LexicalReranker,
    assert_faithfulness_regression,
    compress_context,
    run_evaluation_core,
    run_rag_core,
)
from helpdeskai.retrieval.workflow import SearchHit


class FakeRetrievalEngine:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: object | None = None,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        del top_k, filters, mode
        self.queries.append(query)
        return self.hits


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _build_config(tmp_path: Path) -> RagConfig:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"
    _write_jsonl(chunks_path, [{"chunk_id": "C1", "text": "placeholder"}])
    _write_jsonl(
        golden_path,
        [
            {
                "question_id": "Q1",
                "question": "How do I reset my MQ queue manager password?",
                "expected_answer": "Reset the queue manager password from the admin console.",
                "source": "techqa",
                "doc_id": "D1",
                "intent": "",
                "category": "",
            }
        ],
    )
    return RagConfig(chunks_path=chunks_path, golden_path=golden_path, evaluation_path=tmp_path / "report.json")


def _hits() -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id="C1",
            doc_id="D1",
            score=0.9,
            text=(
                "Reset the queue manager password from the admin console. "
                "Use an administrator account and save the change."
            ),
            source="techqa",
            product="MQ",
            version="9.3",
            category="troubleshooting",
            date="2024-01-01",
        ),
        SearchHit(
            chunk_id="C2",
            doc_id="D2",
            score=0.4,
            text="Review audit settings after you reset any administrator credential.",
            source="techqa",
            product="MQ",
            version="9.3",
            category="operations",
            date="2024-01-01",
        ),
    ]


def test_prompt_variants_are_versioned() -> None:
    assert set(PROMPT_VARIANTS) == {"baseline", "grounded", "concise"}


def test_query_rewriter_removes_duplicate_sentences() -> None:
    rewriter = HeuristicQueryRewriter()
    rewritten = rewriter.rewrite("Reset password? Reset password?   Reset password?")

    assert rewritten == "Reset password?"


def test_compress_context_keeps_most_relevant_sentence() -> None:
    compressed = compress_context(
        "reset password",
        "Open the admin console. Reset the password immediately. Review unrelated metrics later.",
        max_sentences=1,
    )

    assert compressed == "Reset the password immediately."


def test_run_rag_core_orchestrates_retrieval_rerank_and_generation(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    retrieval = FakeRetrievalEngine(_hits())

    payload = run_rag_core(
        config,
        query="How do I reset my MQ queue manager password?",
        retrieval_engine=retrieval,
        query_rewriter=HeuristicQueryRewriter(),
        reranker=LexicalReranker(),
        generator=ExtractiveGenerator(),
    )

    assert retrieval.queries == ["How do I reset my MQ queue manager password?"]
    assert payload["contexts"][0]["chunk_id"] == "C1"
    assert "Reset the queue manager password" in payload["answer"]
    assert [step["step"] for step in payload["steps"]] == [
        "rewrite_query",
        "retrieve",
        "rerank",
        "generate_answer",
    ]


def test_run_rag_core_does_not_require_golden_for_answer(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    config = RagConfig(
        chunks_path=config.chunks_path,
        golden_path=tmp_path / "missing-golden.jsonl",
        evaluation_path=config.evaluation_path,
    )

    payload = run_rag_core(
        config,
        query="How do I reset my MQ queue manager password?",
        retrieval_engine=FakeRetrievalEngine(_hits()),
        query_rewriter=HeuristicQueryRewriter(),
        reranker=LexicalReranker(),
        generator=ExtractiveGenerator(),
    )

    assert payload["answer"]


def test_run_evaluation_core_writes_prompt_comparison_report(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    report = run_evaluation_core(
        config,
        sample_size=1,
        retrieval_engine=FakeRetrievalEngine(_hits()),
        query_rewriter=HeuristicQueryRewriter(),
        reranker=LexicalReranker(),
        generator=ExtractiveGenerator(),
    )

    assert set(report["prompt_variants"]) == {"baseline", "grounded", "concise"}
    assert report["summary"]["best_prompt_variant"] in PROMPT_VARIANTS
    assert config.evaluation_path.exists()


def test_assert_faithfulness_regression_rejects_large_drop() -> None:
    current = {"summary": {"best_faithfulness": 0.80}}
    baseline = {"summary": {"best_faithfulness": 0.90}}

    try:
        assert_faithfulness_regression(current, baseline, max_drop=0.05)
    except ValueError as exc:
        assert "Faithfulness regression exceeds threshold" in str(exc)
    else:
        raise AssertionError("Expected a regression failure")