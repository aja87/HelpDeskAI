"""Continuous evaluation and drift detection utilities."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

JUDGE_MODEL = "claude-sonnet-4-6"

JUDGE_PROMPT = """Evaluate this RAG answer on two criteria from 0 to 1.

Return strict JSON: {"faithfulness": 0.0, "relevancy": 0.0, "reason": "one sentence"}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""


def load_conversation_samples(path: Path, sample_ratio: float, seed: int) -> list[dict[str, Any]]:
    """Load deterministic JSONL samples from production-like conversations."""
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError("sample_ratio must be in ]0, 1]")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(seed)
    return [record for record in records if rng.random() <= sample_ratio]


def judge_answer(
    *,
    question: str,
    answer: str,
    context: str,
    judge=None,
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    """Evaluate one answer with an Anthropic-compatible client or a deterministic fallback."""
    if judge is None:
        if not context.strip():
            return {
                "faithfulness": 0.0,
                "relevancy": 0.4 if answer.strip() else 0.0,
                "reason": "no_context",
            }
        overlap = set(question.lower().split()) & set(answer.lower().split())
        return {
            "faithfulness": 0.8 if answer.strip() and context.strip() else 0.0,
            "relevancy": min(1.0, 0.5 + len(overlap) / 10),
            "reason": "heuristic_offline_judge",
        }

    message = judge.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    context=context,
                    question=question,
                    answer=answer,
                ),
            }
        ],
    )
    text = message.content[0].text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"faithfulness": 0.0, "relevancy": 0.0, "reason": "parse_error"}
    return json.loads(match.group(0))


def detect_drift(
    experiment_name: str,
    baseline_size: int,
    recent_size: int,
    threshold: float,
) -> dict[str, Any]:
    """Compare recent MLflow faithfulness with the previous baseline window."""
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        max_results=baseline_size + recent_size,
    )
    return detect_drift_from_runs(runs, baseline_size, recent_size, threshold)


def detect_drift_from_runs(
    runs: pd.DataFrame,
    baseline_size: int,
    recent_size: int,
    threshold: float,
) -> dict[str, Any]:
    """Pure drift detector used by tests and the MLflow wrapper."""
    if len(runs) < baseline_size + recent_size:
        return {"error": "not_enough_runs", "alert": False}
    if "metrics.faithfulness" not in runs:
        return {"error": "missing_faithfulness", "alert": False}
    recent = runs.head(recent_size)["metrics.faithfulness"].mean()
    baseline = runs.tail(baseline_size)["metrics.faithfulness"].mean()
    delta = recent - baseline
    return {
        "baseline_mean": round(float(baseline), 4),
        "recent_mean": round(float(recent), 4),
        "delta": round(float(delta), 4),
        "threshold": threshold,
        "alert": bool(delta < -threshold),
    }
