from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helpdeskai.observability.continuous_eval import (
    detect_drift_from_runs,
    judge_answer,
    load_conversation_samples,
)
from helpdeskai.observability.finops import Scenario, make_optimized, recommend
from helpdeskai.observability.mlflow_tracking import (
    estimate_rag_cost_usd,
    log_rag_evaluation_run,
)
from helpdeskai.observability.prompt_registry import (
    load_prompt_by_alias,
    promote_prompt_alias,
    register_prompt_versions,
)
from helpdeskai.rag.models import RagContext, RagResult, StageTiming
from scripts.continuous_eval import parse_args as parse_continuous_eval_args
from scripts.detect_eval_drift import parse_args as parse_drift_args
from scripts.finops_dashboard import parse_args as parse_finops_args
from scripts.register_prompts import parse_args as parse_register_prompts_args


def _result() -> RagResult:
    return RagResult(
        question_original="How to configure SAML?",
        question_rewritten="configure SAML",
        answer="Use the admin console [chunk-1]",
        contexts=[
            RagContext(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="SAML configuration uses the admin console.",
                score=0.9,
            )
        ],
        sources=["chunk-1"],
        timings=[StageTiming("retrieval", 10.0), StageTiming("generation", 30.0)],
        model_names={"generator": "claude-haiku-4-5-20251001"},
        prompt_version="strict",
        retrieval_mode="hybrid",
    )


def test_finops_optimized_scenario_reduces_llm_cost_and_exports_rows() -> None:
    baseline = Scenario("Scale", requests_per_month=100_000, infra_usd_per_month=200)
    optimized = make_optimized(baseline)

    baseline_cost = baseline.effective_cost()
    optimized_cost = optimized.effective_cost()

    assert optimized_cost["llm_cost_usd"] < baseline_cost["llm_cost_usd"]
    assert optimized.to_row(variant="optimized")["variant"] == "optimized"
    assert "Scale" in recommend([baseline])[0]


def test_estimate_rag_cost_is_deterministic() -> None:
    cost = estimate_rag_cost_usd([_result()], model="claude-haiku-4-5-20251001")

    assert cost > 0
    assert cost == estimate_rag_cost_usd([_result()], model="claude-haiku-4-5-20251001")


def test_mlflow_rag_evaluation_logging_uses_temp_store(tmp_path: Path) -> None:
    import mlflow

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifact = tmp_path / "ragas.csv"
    artifact.write_text("faithfulness\n0.9\n", encoding="utf-8")
    golden = tmp_path / "questions.jsonl"
    golden.write_text('{"question":"q"}\n', encoding="utf-8")

    run_id = log_rag_evaluation_run(
        tracking_uri=tracking_uri,
        experiment="test-rag-eval",
        run_name="unit",
        params={"generator_model": "claude-haiku-4-5-20251001", "prompt_version": "strict"},
        scores={"strict.faithfulness": 0.9},
        results=[_result()],
        artifact_paths=[artifact],
        golden_path=golden,
    )

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["strict.faithfulness"] == 0.9
    assert run.data.metrics["latency_p95_ms"] == 40.0


def test_prompt_registry_aliases_with_temp_mlflow_store(tmp_path: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    run_ids = register_prompt_versions(
        {"strict": "Strict prompt", "concise": "Concise prompt"},
        {"production": "strict", "dev": "concise"},
        prompt_name="unit-prompt",
        experiment="unit-prompts",
    )

    loaded = load_prompt_by_alias("unit-prompt", "production", experiment="unit-prompts")

    assert set(run_ids) == {"strict", "concise"}
    assert loaded == ("strict", "Strict prompt")

    promote_prompt_alias("unit-prompt", "concise", "production", experiment="unit-prompts")
    assert load_prompt_by_alias("unit-prompt", "production", experiment="unit-prompts") == (
        "concise",
        "Concise prompt",
    )


def test_conversation_sampling_is_deterministic(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations.jsonl"
    conversations.write_text(
        "\n".join(json.dumps({"conversation_id": f"c{i}"}) for i in range(20)) + "\n",
        encoding="utf-8",
    )

    first = load_conversation_samples(conversations, sample_ratio=0.25, seed=7)
    second = load_conversation_samples(conversations, sample_ratio=0.25, seed=7)

    assert first == second
    assert 0 < len(first) < 20


def test_offline_judge_and_drift_detection() -> None:
    score = judge_answer(
        question="SAML setup",
        answer="SAML setup uses admin",
        context="SAML setup",
    )
    assert score["faithfulness"] > 0

    runs = pd.DataFrame(
        {
            "metrics.faithfulness": [
                0.75,
                0.76,
                0.77,
                0.88,
                0.89,
                0.90,
            ]
        }
    )
    drift = detect_drift_from_runs(runs, baseline_size=3, recent_size=3, threshold=0.05)

    assert drift["alert"] is True
    assert drift["delta"] < -0.05


def test_phase7_script_arg_parsers() -> None:
    assert parse_register_prompts_args(["--promote-from-eval"]).promote_from_eval is True
    assert parse_finops_args(["--csv", "out.csv"]).csv == Path("out.csv")
    assert parse_continuous_eval_args(["--sample-ratio", "0.2"]).sample_ratio == 0.2
    assert parse_drift_args(["--threshold", "0.1"]).threshold == 0.1
