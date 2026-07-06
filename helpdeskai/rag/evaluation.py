"""Ragas evaluation utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from helpdeskai.ingestion.io import read_jsonl, write_jsonl
from helpdeskai.rag.models import RagConfig, RagResult
from helpdeskai.rag.pipeline import AdvancedRagPipeline, result_to_ragas_row
from helpdeskai.rag.prompts import PROMPT_VARIANTS

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_ragas_cases(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load retrieval-eligible TechQA golden cases for Ragas."""
    cases = [
        record
        for record in read_jsonl(path)
        if record.get("retrieval_eligible") is True
        and record.get("source") == "techqa"
        and record.get("document_id")
        and record.get("reference_answer")
    ]
    return cases[:limit] if limit is not None else cases


def build_results_for_prompt(
    cases: Sequence[dict[str, Any]],
    *,
    config: RagConfig,
    pipeline: AdvancedRagPipeline | None = None,
) -> list[RagResult]:
    """Run the RAG pipeline for one prompt variant over evaluation cases."""
    runner = pipeline or AdvancedRagPipeline(config=config)
    return [runner.run(str(case["question"])) for case in cases]


def build_ragas_rows(
    cases: Sequence[dict[str, Any]],
    results: Sequence[RagResult],
) -> list[dict[str, Any]]:
    """Create Ragas rows from aligned golden cases and RAG results."""
    return [
        result_to_ragas_row(result, ground_truth=str(case["reference_answer"]))
        for case, result in zip(cases, results, strict=True)
    ]


def evaluate_ragas_rows(
    rows: Sequence[dict[str, Any]],
    *,
    judge_model: str,
    embedding_model: str = "intfloat/multilingual-e5-small",
) -> pd.DataFrame:
    """Evaluate rows with Ragas and return the detailed dataframe."""
    from datasets import Dataset
    from langchain_anthropic import ChatAnthropic
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_list(list(rows))
    judge_llm = LangchainLLMWrapper(ChatAnthropic(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=embedding_model)
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
    )
    return result.to_pandas()


def summarize_scores(dataframe: pd.DataFrame) -> dict[str, float]:
    """Return mean score for each expected Ragas metric."""
    return {
        metric: round(float(dataframe[metric].dropna().mean()), 4)
        for metric in METRIC_NAMES
        if metric in dataframe
    }


def choose_best_prompt(summaries: dict[str, dict[str, float]]) -> str:
    """Choose best prompt by faithfulness first, then average score."""
    return max(
        summaries,
        key=lambda name: (
            summaries[name].get("faithfulness", 0.0),
            sum(summaries[name].values()) / max(len(summaries[name]), 1),
        ),
    )


def write_rag_results(path: Path, results: Sequence[RagResult]) -> Path:
    """Write RAG run results as JSONL."""
    write_jsonl(path, [result.to_dict() for result in results])
    return path


def write_comparison(report_dir: Path, summaries: dict[str, dict[str, float]]) -> tuple[Path, Path]:
    """Write Markdown and JSON prompt comparison artifacts."""
    report_dir.mkdir(parents=True, exist_ok=True)
    best = choose_best_prompt(summaries)
    json_path = report_dir / "ragas_comparison.json"
    markdown_path = report_dir / "ragas_comparison.md"
    json_path.write_text(
        json.dumps(
            {"best_prompt": best, "scores": summaries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RAG prompt comparison",
        "",
        f"Best prompt: `{best}`",
        "",
        "| Metric | " + " | ".join(PROMPT_VARIANTS) + " | Winner |",
        "| --- | " + " | ".join("---:" for _ in PROMPT_VARIANTS) + " | --- |",
    ]
    for metric in METRIC_NAMES:
        values = {name: summaries[name].get(metric, 0.0) for name in PROMPT_VARIANTS}
        winner = max(values, key=values.get)
        lines.append(
            f"| {metric} | "
            + " | ".join(f"{values[name]:.4f}" for name in PROMPT_VARIANTS)
            + f" | {winner} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def prompt_names(selected: str) -> Iterable[str]:
    """Resolve selected prompt option."""
    if selected == "all":
        return PROMPT_VARIANTS.keys()
    if selected not in PROMPT_VARIANTS:
        allowed = ", ".join(["all", *PROMPT_VARIANTS])
        raise ValueError(f"unknown prompt '{selected}'. Expected one of: {allowed}")
    return [selected]
