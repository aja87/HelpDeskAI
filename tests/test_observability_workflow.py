from __future__ import annotations

import json

from pathlib import Path

from helpdeskai.observability.config import ObservabilityConfig
from helpdeskai.observability.io_utils import read_json
from helpdeskai.observability.workflow import (
    UsageRecord,
    build_finops_dashboard,
    export_langfuse_trace,
    run_continuous_evaluation,
    run_observability_core,
    track_eval_runs,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _build_config(tmp_path: Path) -> ObservabilityConfig:
    reports_dir = tmp_path / "reports"
    return ObservabilityConfig(
        reports_dir=reports_dir,
        traces_dir=reports_dir / "traces",
        golden_path=tmp_path / "golden.jsonl",
        conversations_path=reports_dir / "simulated_conversations.jsonl",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}",
        monthly_budget_usd=40.0,
        continuous_sample_ratio=0.2,
        seed=7,
    )


def test_track_eval_runs_logs_three_prompt_versions(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    _write_jsonl(config.golden_path, [{"question": "q", "expected_answer": "a"}])

    payload = track_eval_runs(config)

    assert set(payload["run_ids"]) == {"v1", "v2", "v3"}
    assert payload["winner_version"] in {"v1", "v2", "v3"}
    assert payload["metrics"]["v3"]["faithfulness"] >= payload["metrics"]["v1"]["faithfulness"]


def test_export_langfuse_trace_writes_trace_artifact(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    config.traces_dir.mkdir(parents=True, exist_ok=True)

    payload = export_langfuse_trace(
        config,
        session_id="session-test",
        query="How to reset password?",
        answer="Use admin console security panel.",
        tool_calls=[{"name": "retrieve", "latency_ms": 23.4, "cost_usd": 0.0002}],
    )

    trace_path = Path(payload["trace_path"])
    assert trace_path.exists()
    assert payload["event_count"] == 3


def test_build_finops_dashboard_flags_budget_overrun(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    payload = build_finops_dashboard(
        config,
        [
            UsageRecord(
                use_case="rag_answers",
                requests=120_000,
                model="claude-sonnet-4-6",
                input_tokens=4_000,
                output_tokens=500,
            )
        ],
    )

    assert payload["budget_exceeded"]
    assert "rag_answers" in payload["cost_by_use_case"]
    assert (config.reports_dir / "finops_dashboard.json").exists()


def test_run_continuous_evaluation_samples_and_reports(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    rows = [
        {
            "conversation_id": f"conv-{i}",
            "query": "How to reset MQ password?",
            "expected_answer": "Reset from admin console.",
            "answer": "Reset from admin console.",
        }
        for i in range(20)
    ]
    _write_jsonl(config.conversations_path, rows)

    payload = run_continuous_evaluation(config)

    assert payload["sample_size"] == 4
    assert payload["summary"]["faithfulness_mean"] >= 0.9


def test_run_observability_core_all_generates_full_report(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    _write_jsonl(config.golden_path, [{"question": "q", "expected_answer": "a"}])
    _write_jsonl(
        config.conversations_path,
        [
            {
                "conversation_id": "conv-1",
                "query": "How to reset MQ password?",
                "expected_answer": "Reset from admin console.",
                "answer": "Reset from admin console.",
            },
            {
                "conversation_id": "conv-2",
                "query": "How to reset MQ password?",
                "expected_answer": "Reset from admin console.",
                "answer": "Please reboot the server.",
            },
        ],
    )

    payload = run_observability_core(config, action="all")

    assert "tracking" in payload
    assert "prompt_registry" in payload
    assert "model_registry" in payload
    assert "trace" in payload
    assert "finops" in payload
    assert "continuous_evaluation" in payload

    final_report = read_json(config.reports_dir / "observability_run_report.json")
    assert final_report["action"] == "all"
