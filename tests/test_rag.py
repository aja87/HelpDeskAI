from __future__ import annotations

import pytest

from helpdeskai.rag.evaluation import load_ragas_cases
from helpdeskai.rag.models import RagConfig
from helpdeskai.rag.pipeline import AdvancedRagPipeline, compress_contexts
from helpdeskai.rag.prompts import PROMPT_VARIANTS, get_prompt_variant
from helpdeskai.retrieval.models import SearchMode, SearchResult
from scripts.evaluate_rag import parse_args as parse_evaluate_args
from scripts.run_rag import parse_args as parse_run_args


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
        self.prompts.append(prompt)
        if "Reformulation" in prompt:
            return "rewritten saml query"
        return "Answer grounded in context [chunk-b]"


class FakeSearchEngine:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int, mode: str):
        self.queries.append(query)
        return [
            result("chunk-a", "doc-a", "low relevance content", 0.2),
            result("chunk-b", "doc-b", "SAML authentication setup", 0.9),
        ][:top_k]


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(self, query: str, candidates, *, top_k: int):
        return sorted(candidates, key=lambda item: item.chunk_id, reverse=True)[:top_k]


def result(chunk_id: str, document_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        score=score,
        mode=SearchMode.HYBRID,
        metadata={"product": "NovaCloud"},
        source_scores={"hybrid": score},
    )


def test_pipeline_uses_rewritten_query_and_returns_structured_result() -> None:
    llm = FakeLlm()
    search = FakeSearchEngine()
    pipeline = AdvancedRagPipeline(
        config=RagConfig(final_k=1, max_context_chars=200),
        llm=llm,
        search_engine=search,
        reranker=FakeReranker(),
    )

    output = pipeline.run("How to configure login?")

    assert search.queries == ["rewritten saml query"]
    assert output.question_rewritten == "rewritten saml query"
    assert output.answer == "Answer grounded in context [chunk-b]"
    assert output.sources == ["chunk-b"]
    assert output.contexts[0].document_id == "doc-b"
    assert output.model_names["reranker"] == "fake-reranker"
    assert len(output.timings) == 5
    assert output.to_dict()["prompt_version"] == "strict"


def test_context_compression_respects_budget() -> None:
    contexts = [
        result("one", "doc-1", "a" * 50, 0.9),
        result("two", "doc-2", "b" * 50, 0.8),
    ]

    compressed = compress_contexts(contexts, max_chars=60)

    assert len(compressed) == 2
    assert len("".join(context.content for context in compressed)) == 60
    assert compressed[1].content == "b" * 10


def test_prompt_variants_are_executable_and_validated() -> None:
    for prompt in PROMPT_VARIANTS.values():
        text = prompt("Question?", "[chunk-1] Context")
        assert "Question?" in text
        assert "[chunk_id]" in text

    with pytest.raises(ValueError, match="unknown prompt"):
        get_prompt_variant("missing")


def test_load_ragas_cases_filters_to_retrieval_eligible_techqa(tmp_path) -> None:
    golden = tmp_path / "questions.jsonl"
    golden.write_text(
        "\n".join(
            [
                (
                    '{"source":"techqa","retrieval_eligible":true,"document_id":"doc-1",'
                    '"question":"q","reference_answer":"a"}'
                ),
                (
                    '{"source":"bitext","retrieval_eligible":false,'
                    '"question":"q","reference_answer":"a"}'
                ),
                (
                    '{"source":"techqa","retrieval_eligible":true,'
                    '"question":"q","reference_answer":"a"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_ragas_cases(golden)

    assert len(cases) == 1
    assert cases[0]["document_id"] == "doc-1"


def test_script_arg_parsers_support_help_related_defaults() -> None:
    run_args = parse_run_args(["--question", "hello", "--prompt", "concise"])
    eval_args = parse_evaluate_args(["--prompt", "strict", "--limit", "2"])

    assert run_args.question == ["hello"]
    assert run_args.prompt == "concise"
    assert eval_args.prompt == "strict"
    assert eval_args.limit == 2
