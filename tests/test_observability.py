from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helpdeskai.observability.continuous_eval import (
    detect_drift_from_runs,
    judge_answer,
    load_conversation_samples,
)
from helpdeskai.observability.finops import Scenario, make_current_poc, make_optimized, recommend
from helpdeskai.observability.mlflow_model import RagChainPyfuncModel
from helpdeskai.observability.mlflow_tracking import (
    estimate_rag_cost_usd,
    log_rag_evaluation_run,
)
from helpdeskai.observability.prompt_registry import (
    load_prompt_by_alias,
    promote_prompt_alias,
    register_prompt_versions,
)
from helpdeskai.rag.models import RagConfig, RagContext, RagResult, StageTiming
from scripts.continuous_eval import parse_args as parse_continuous_eval_args
from scripts.detect_eval_drift import parse_args as parse_drift_args
from scripts.finops_dashboard import parse_args as parse_finops_args
from scripts.register_prompts import parse_args as parse_register_prompts_args
from scripts.register_rag_model import parse_args as parse_register_rag_model_args


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


class FakeRagPipeline:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def run(self, question: str) -> RagResult:
        base = _result()
        return RagResult(
            question_original=question,
            question_rewritten=f"rewritten {question}",
            answer=base.answer,
            contexts=base.contexts,
            sources=base.sources,
            timings=base.timings,
            model_names=base.model_names,
            prompt_version=self.config.prompt_version,
            retrieval_mode=self.config.retrieval_mode,
        )


def test_finops_optimized_scenario_reduces_llm_cost_and_exports_rows() -> None:
    baseline = Scenario("Scale", requests_per_month=100_000, infra_usd_per_month=200)
    current = make_current_poc(baseline)
    optimized = make_optimized(baseline)

    baseline_cost = baseline.effective_cost()
    current_cost = current.effective_cost()
    optimized_cost = optimized.effective_cost()

    assert current_cost["total_usd"] < baseline_cost["total_usd"]
    assert optimized_cost["llm_cost_usd"] < baseline_cost["llm_cost_usd"]
    assert optimized_cost["total_usd"] < current_cost["total_usd"]
    assert current.to_row(variant="current_poc")["variant"] == "current_poc"
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


def test_rag_chain_pyfunc_predicts_structured_rows_without_external_services() -> None:
    model = RagChainPyfuncModel(
        RagConfig(prompt_version="concise", retrieval_mode="sparse"),
        pipeline_factory=FakeRagPipeline,
    )

    output = model.predict(None, pd.DataFrame({"question": ["q1", "q2"]}))

    assert list(output["question"]) == ["q1", "q2"]
    assert output.loc[0, "question_rewritten"] == "rewritten q1"
    assert output.loc[0, "prompt_version"] == "concise"
    assert output.loc[0, "retrieval_mode"] == "sparse"
    assert output.loc[0, "sources"] == ["chunk-1"]

    dict_output = model.predict(None, {"question": ["q3", "q4"]})
    assert list(dict_output["question"]) == ["q3", "q4"]


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


def test_register_rag_model_arg_parser_defaults_to_production_alias() -> None:
    args = parse_register_rag_model_args(["--model-name", "unit-rag"])

    assert args.model_name == "unit-rag"
    assert args.alias == "production"
    assert args.mode == "hybrid"
