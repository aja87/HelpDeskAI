"""MLflow tracking helpers for RAG evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from helpdeskai.observability.finops import PRICING_USD_PER_MILLION_TOKENS
from helpdeskai.rag.models import RagResult


def configure_mlflow(tracking_uri: str, experiment: str) -> None:
    """Configure MLflow tracking for a script."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def estimate_rag_cost_usd(
    results: Sequence[RagResult],
    *,
    model: str,
    input_token_multiplier: float = 0.25,
    output_token_multiplier: float = 0.25,
) -> float:
    """Estimate total RAG generation cost from result text lengths.

    The estimate is intentionally deterministic for reports and tests. Real provider billing
    should come from Langfuse/MLflow traces when available.
    """
    pricing = PRICING_USD_PER_MILLION_TOKENS.get(
        model,
        PRICING_USD_PER_MILLION_TOKENS["claude-haiku-4-5"],
    )
    total = 0.0
    for result in results:
        context_chars = sum(len(context.content) for context in result.contexts)
        input_tokens = int((len(result.question_original) + context_chars) * input_token_multiplier)
        output_tokens = int(len(result.answer) * output_token_multiplier)
        total += input_tokens / 1_000_000 * pricing["input"]
        total += output_tokens / 1_000_000 * pricing["output"]
    return round(total, 6)


def _latency_metrics(results: Sequence[RagResult]) -> dict[str, float]:
    totals = [sum(timing.duration_ms for timing in result.timings) for result in results]
    if not totals:
        return {"latency_avg_ms": 0.0, "latency_p95_ms": 0.0}
    sorted_totals = sorted(totals)
    p95_index = min(len(sorted_totals) - 1, int(len(sorted_totals) * 0.95))
    return {
        "latency_avg_ms": round(mean(totals), 2),
        "latency_p95_ms": round(sorted_totals[p95_index], 2),
    }


def log_rag_evaluation_run(
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    scores: Mapping[str, float],
    results: Sequence[RagResult],
    artifact_paths: Sequence[Path],
    golden_path: Path | None = None,
) -> str:
    """Log one RAG evaluation run to MLflow and return its run id."""
    import mlflow

    configure_mlflow(tracking_uri, experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics({key: float(value) for key, value in scores.items()})
        mlflow.log_metrics(_latency_metrics(results))
        generator_model = str(params.get("generator_model", "claude-haiku-4-5"))
        cost_total = estimate_rag_cost_usd(results, model=generator_model)
        mlflow.log_metric("cost_total_usd", cost_total)
        mlflow.log_metric("cost_per_query_usd", cost_total / max(len(results), 1))
        if golden_path and golden_path.exists():
            mlflow.log_artifact(str(golden_path), artifact_path="golden")
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="reports")
        return run.info.run_id
