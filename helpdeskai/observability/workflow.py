from __future__ import annotations

import json
import logging
import random
import statistics
import time

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

from .config import ObservabilityConfig
from .io_utils import read_jsonl, write_json, write_jsonl


PROMPT_VERSIONS = {
    "v1": "You are a support assistant. Answer the user question: {question}",
    "v2": (
        "You are a support assistant. Answer ONLY from provided context.\\n"
        "Context:\\n{context}\\n\\nQuestion:\\n{question}"
    ),
    "v3": (
        "You are a senior support assistant. Ground every statement in context and cite [chunk_id].\\n"
        "If the answer is missing, state it clearly.\\n\\nContext:\\n{context}\\n\\nQuestion:\\n{question}"
    ),
}

MODEL_PRICING_PER_1M_TOKENS_USD = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass(slots=True)
class UsageRecord:
    """Per-use-case usage for FinOps computations."""

    use_case: str
    requests: int
    model: str
    input_tokens: int
    output_tokens: int


class HelpDeskRagPyFunc(PythonModel):
    """Minimal pyfunc wrapper for RAG chain registration demos."""

    def predict(self, context, model_input, params=None):
        del context, params
        if hasattr(model_input, "to_dict"):
            payload = model_input.to_dict(orient="records")
        else:
            payload = list(model_input)

        outputs: list[str] = []
        for row in payload:
            query = str(row.get("query", "")).strip()
            context_text = str(row.get("context", "")).strip()
            if context_text:
                outputs.append(f"Grounded answer: {context_text.split('. ')[0].strip()}")
            else:
                outputs.append(f"Insufficient context to answer safely for query: {query}")
        return outputs


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if token]


def _simulate_eval_metrics(prompt_version: str, seed: int) -> dict[str, float]:
    rng = random.Random(f"{seed}:{prompt_version}")
    base = {"v1": 0.72, "v2": 0.84, "v3": 0.91}[prompt_version]
    faithfulness = max(0.0, min(1.0, base + rng.uniform(-0.015, 0.015)))
    relevancy = max(0.0, min(1.0, base + 0.04 + rng.uniform(-0.015, 0.015)))
    latency_ms = max(150.0, 900 + rng.uniform(-120, 120))
    cost_usd = max(0.0001, 0.005 + rng.uniform(-0.001, 0.001))
    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "latency_p95_ms": round(latency_ms, 2),
        "cost_per_query_usd": round(cost_usd, 5),
    }


def _clear_prompt_alias(client: MlflowClient, experiment_id: str, alias: str) -> None:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`alias.{alias}` = 'current'",
        max_results=100,
    )
    for run in runs:
        client.delete_tag(run.info.run_id, f"alias.{alias}")


def track_eval_runs(config: ObservabilityConfig) -> dict[str, Any]:
    """Log RAG evaluation runs to MLflow with params, metrics, and artifacts."""

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.eval_experiment)

    run_ids: dict[str, str] = {}
    metrics_by_version: dict[str, dict[str, float]] = {}

    for version, prompt_text in PROMPT_VERSIONS.items():
        with mlflow.start_run(run_name=f"eval_{version}") as run:
            metrics = _simulate_eval_metrics(version, config.seed)
            mlflow.log_params(
                {
                    "prompt_version": version,
                    "model": config.generator_model,
                    "temperature": 0,
                    "retrieval_mode": config.retrieval_mode,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_text(prompt_text, f"prompts/{version}.txt")
            if config.golden_path.exists():
                mlflow.log_artifact(str(config.golden_path), artifact_path="datasets")
            run_ids[version] = run.info.run_id
            metrics_by_version[version] = metrics

    winner_version = max(metrics_by_version, key=lambda key: metrics_by_version[key]["faithfulness"])
    return {
        "run_ids": run_ids,
        "metrics": metrics_by_version,
        "winner_version": winner_version,
    }


def register_prompts(config: ObservabilityConfig, eval_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Register prompt versions and promote aliases in MLflow experiments."""

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.prompt_experiment)

    run_ids: dict[str, str] = {}
    for version, prompt_text in PROMPT_VERSIONS.items():
        with mlflow.start_run(run_name=f"register_{version}") as run:
            mlflow.log_params({"prompt_name": config.prompt_name, "version": version})
            mlflow.log_text(prompt_text, f"{config.prompt_name}/{version}.txt")
            run_ids[version] = run.info.run_id

    winner_version = (
        eval_payload["winner_version"]
        if eval_payload is not None
        else max(PROMPT_VERSIONS.keys())
    )

    client = MlflowClient(tracking_uri=config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.prompt_experiment)
    if experiment is None:
        raise RuntimeError("Prompt experiment was not created")

    _clear_prompt_alias(client, experiment.experiment_id, "dev")
    _clear_prompt_alias(client, experiment.experiment_id, "staging")
    _clear_prompt_alias(client, experiment.experiment_id, "production")

    client.set_tag(run_ids[config.prompt_registry_dev_version], "alias.dev", "current")
    client.set_tag(run_ids[config.prompt_registry_staging_version], "alias.staging", "current")
    client.set_tag(run_ids[winner_version], "alias.production", "current")

    return {
        "run_ids": run_ids,
        "winner_version": winner_version,
        "aliases": {
            "dev": config.prompt_registry_dev_version,
            "staging": config.prompt_registry_staging_version,
            "production": winner_version,
        },
    }


def register_rag_model(config: ObservabilityConfig, prompt_payload: dict[str, Any]) -> dict[str, Any]:
    """Register a lightweight RAG pyfunc model and promote it to production alias."""

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.model_experiment)

    input_example = [{"query": "How do I reset admin password?", "context": "Use admin console security panel."}]
    signature = infer_signature(
        model_input=input_example,
        model_output=["Grounded answer: Use admin console security panel."],
    )

    with mlflow.start_run(run_name="register_rag_pyfunc") as run:
        mlflow.log_params(
            {
                "model_name": config.registered_model_name,
                "prompt_production_version": prompt_payload["aliases"]["production"],
                "generator_model": config.generator_model,
                "retrieval_mode": config.retrieval_mode,
            }
        )
        mlflow.pyfunc.log_model(
            artifact_path="rag_chain",
            python_model=HelpDeskRagPyFunc(),
            input_example=input_example,
            signature=signature,
            registered_model_name=config.registered_model_name,
        )
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=config.tracking_uri)
    versions = client.search_model_versions(f"name='{config.registered_model_name}'")
    if not versions:
        raise RuntimeError("No model version found after registration")
    latest = max(versions, key=lambda item: int(item.version))
    client.set_registered_model_alias(
        name=config.registered_model_name,
        alias=config.production_alias,
        version=latest.version,
    )

    return {
        "run_id": run_id,
        "model_name": config.registered_model_name,
        "model_version": int(latest.version),
        "alias": config.production_alias,
    }


def export_langfuse_trace(
    config: ObservabilityConfig,
    *,
    session_id: str,
    query: str,
    answer: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export a deterministic trace artifact compatible with local review."""

    tool_calls = tool_calls or []
    events: list[dict[str, Any]] = []

    start = time.perf_counter()
    events.append(
        {
            "timestamp": _utc_now_iso(),
            "session_id": session_id,
            "span": "user_query",
            "latency_ms": 2.0,
            "cost_usd": 0.0,
            "payload": {"query": query},
        }
    )

    for index, call in enumerate(tool_calls, start=1):
        events.append(
            {
                "timestamp": _utc_now_iso(),
                "session_id": session_id,
                "span": f"tool_{index}",
                "latency_ms": float(call.get("latency_ms", 35.0)),
                "cost_usd": float(call.get("cost_usd", 0.0002)),
                "payload": call,
            }
        )

    total_latency_ms = round((time.perf_counter() - start) * 1000 + 120.0, 2)
    estimated_cost = round(0.001 + 0.0003 * len(tool_calls), 5)
    events.append(
        {
            "timestamp": _utc_now_iso(),
            "session_id": session_id,
            "span": "assistant_answer",
            "latency_ms": total_latency_ms,
            "cost_usd": estimated_cost,
            "payload": {"answer": answer},
        }
    )

    output_path = config.traces_dir / f"trace_{session_id}.jsonl"
    write_jsonl(output_path, events)

    return {
        "session_id": session_id,
        "trace_path": str(output_path),
        "event_count": len(events),
        "total_estimated_cost_usd": round(sum(event["cost_usd"] for event in events), 5),
    }


def build_finops_dashboard(
    config: ObservabilityConfig,
    usage_rows: list[UsageRecord],
) -> dict[str, Any]:
    """Compute cost by use-case and flag monthly budget overruns."""

    per_use_case: dict[str, dict[str, float]] = {}

    for row in usage_rows:
        pricing = MODEL_PRICING_PER_1M_TOKENS_USD.get(
            row.model,
            MODEL_PRICING_PER_1M_TOKENS_USD["claude-haiku-4-5-20251001"],
        )
        input_cost = (row.requests * row.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (row.requests * row.output_tokens / 1_000_000) * pricing["output"]
        total = input_cost + output_cost

        entry = per_use_case.setdefault(
            row.use_case,
            {
                "requests": 0.0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "llm_cost_usd": 0.0,
            },
        )
        entry["requests"] += float(row.requests)
        entry["input_tokens"] += float(row.requests * row.input_tokens)
        entry["output_tokens"] += float(row.requests * row.output_tokens)
        entry["llm_cost_usd"] += total

    total_cost = round(sum(item["llm_cost_usd"] for item in per_use_case.values()), 4)
    projected_monthly_cost = round(total_cost * 1.05, 4)
    budget_exceeded = projected_monthly_cost > config.monthly_budget_usd

    summary = {
        "generated_at": _utc_now_iso(),
        "monthly_budget_usd": config.monthly_budget_usd,
        "projected_monthly_cost_usd": projected_monthly_cost,
        "budget_exceeded": budget_exceeded,
        "cost_by_use_case": {
            name: {
                "requests": int(values["requests"]),
                "input_tokens": int(values["input_tokens"]),
                "output_tokens": int(values["output_tokens"]),
                "llm_cost_usd": round(values["llm_cost_usd"], 4),
            }
            for name, values in per_use_case.items()
        },
    }

    write_json(config.reports_dir / "finops_dashboard.json", summary)
    return summary


def _evaluate_llm_judge_like(query: str, expected_answer: str, answer: str) -> dict[str, float]:
    query_tokens = set(_tokenize(query))
    expected_tokens = set(_tokenize(expected_answer))
    answer_tokens = set(_tokenize(answer))

    faithfulness = len(answer_tokens & expected_tokens) / len(answer_tokens) if answer_tokens else 0.0
    relevancy = len(answer_tokens & query_tokens) / len(query_tokens) if query_tokens else 0.0

    return {
        "faithfulness": round(min(1.0, max(0.0, faithfulness)), 4),
        "answer_relevancy": round(min(1.0, max(0.0, relevancy)), 4),
    }


def run_continuous_evaluation(config: ObservabilityConfig) -> dict[str, Any]:
    """Sample simulated production conversations, judge, and log back to MLflow."""

    rows = read_jsonl(config.conversations_path)
    if not rows:
        raise ValueError("No conversation rows available for continuous evaluation")

    sample_size = max(1, int(len(rows) * config.continuous_sample_ratio))
    rng = random.Random(config.seed)
    sampled = rng.sample(rows, min(sample_size, len(rows)))

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.continuous_eval_experiment)

    scored_rows: list[dict[str, Any]] = []
    faithfulness_values: list[float] = []
    relevancy_values: list[float] = []

    for row in sampled:
        query = str(row.get("query", ""))
        expected = str(row.get("expected_answer", ""))
        answer = str(row.get("answer", ""))
        metrics = _evaluate_llm_judge_like(query, expected, answer)

        with mlflow.start_run(run_name=f"continuous_eval_{row.get('conversation_id', 'unknown')}"):
            mlflow.log_params(
                {
                    "conversation_id": str(row.get("conversation_id", "")),
                    "judge_model": config.judge_model,
                }
            )
            mlflow.log_metrics(metrics)

        merged = dict(row)
        merged.update(metrics)
        scored_rows.append(merged)
        faithfulness_values.append(metrics["faithfulness"])
        relevancy_values.append(metrics["answer_relevancy"])

    report = {
        "generated_at": _utc_now_iso(),
        "sample_size": len(scored_rows),
        "input_population": len(rows),
        "summary": {
            "faithfulness_mean": round(statistics.fmean(faithfulness_values), 4),
            "answer_relevancy_mean": round(statistics.fmean(relevancy_values), 4),
        },
        "samples": scored_rows,
    }

    write_json(config.reports_dir / "continuous_evaluation_report.json", report)
    return report


def _default_usage_rows() -> list[UsageRecord]:
    return [
        UsageRecord(
            use_case="rag_answers",
            requests=20_000,
            model="claude-haiku-4-5-20251001",
            input_tokens=2500,
            output_tokens=280,
        ),
        UsageRecord(
            use_case="agent_escalation",
            requests=2_000,
            model="claude-sonnet-4-6",
            input_tokens=4200,
            output_tokens=450,
        ),
        UsageRecord(
            use_case="offline_eval",
            requests=1_500,
            model="gpt-4o-mini",
            input_tokens=3800,
            output_tokens=160,
        ),
    ]


def _default_conversations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(1, 31):
        rows.append(
            {
                "conversation_id": f"conv-{index}",
                "query": "How do I reset MQ queue manager password?",
                "expected_answer": "Reset the queue manager password from the admin console.",
                "answer": (
                    "Reset the queue manager password from the admin console and validate access."
                    if index % 6
                    else "I think rebooting the server should solve it."
                ),
            }
        )
    return rows


def run_observability_core(config: ObservabilityConfig, *, action: str) -> dict[str, Any]:
    """Execute observability actions as an orchestrated workflow."""

    config.validate()
    if action not in {"track", "prompts", "model", "trace", "finops", "continuous-eval", "all"}:
        raise ValueError("Unsupported action")

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.traces_dir.mkdir(parents=True, exist_ok=True)

    if not config.conversations_path.exists():
        write_jsonl(config.conversations_path, _default_conversations())

    outcome: dict[str, Any] = {"action": action, "generated_at": _utc_now_iso()}

    eval_payload: dict[str, Any] | None = None
    prompt_payload: dict[str, Any] | None = None

    selected = (
        ["track", "prompts", "model", "trace", "finops", "continuous-eval"]
        if action == "all"
        else [action]
    )

    if "track" in selected:
        eval_payload = track_eval_runs(config)
        outcome["tracking"] = eval_payload
        logging.info("Logged MLflow evaluation runs for prompt variants")

    if "prompts" in selected:
        prompt_payload = register_prompts(config, eval_payload=eval_payload)
        outcome["prompt_registry"] = prompt_payload
        logging.info("Updated prompt aliases dev/staging/production")

    if "model" in selected:
        if prompt_payload is None:
            prompt_payload = register_prompts(config, eval_payload=eval_payload)
            outcome["prompt_registry"] = prompt_payload
        model_payload = register_rag_model(config, prompt_payload)
        outcome["model_registry"] = model_payload
        logging.info("Registered and promoted pyfunc model alias=%s", config.production_alias)

    if "trace" in selected:
        trace_payload = export_langfuse_trace(
            config,
            session_id=f"session-{config.seed}",
            query="How do I reset MQ queue manager password?",
            answer="Reset the queue manager password from the admin console and audit the change.",
            tool_calls=[
                {"name": "retrieve_chunks", "latency_ms": 32.1, "cost_usd": 0.0002},
                {"name": "rerank", "latency_ms": 18.0, "cost_usd": 0.0001},
            ],
        )
        outcome["trace"] = trace_payload
        logging.info("Wrote trace artifact for session=%s", trace_payload["session_id"])

    if "finops" in selected:
        finops_payload = build_finops_dashboard(config, _default_usage_rows())
        outcome["finops"] = finops_payload
        logging.info("Computed FinOps dashboard with budget status=%s", finops_payload["budget_exceeded"])

    if "continuous-eval" in selected:
        continuous_payload = run_continuous_evaluation(config)
        outcome["continuous_evaluation"] = continuous_payload
        logging.info("Logged continuous evaluation sample_size=%s", continuous_payload["sample_size"])

    write_json(config.reports_dir / "observability_run_report.json", outcome)
    return outcome
