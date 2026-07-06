from __future__ import annotations

import importlib.util
import json
import sys

from pathlib import Path


def _load_script_module() -> object:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_observability.py"
    spec = importlib.util.spec_from_file_location("run_observability_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load run_observability module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_supports_action_and_overrides(monkeypatch: object, tmp_path: Path) -> None:
    module = _load_script_module()

    args = [
        "run_observability.py",
        "finops",
        "--monthly-budget",
        "75",
        "--continuous-sample-ratio",
        "0.2",
        "--reports-dir",
        str(tmp_path / "reports"),
    ]
    monkeypatch.setattr(sys, "argv", args)

    parsed = module.parse_args()

    assert parsed.action == "finops"
    assert parsed.monthly_budget == 75.0
    assert parsed.continuous_sample_ratio == 0.2
    assert parsed.reports_dir == tmp_path / "reports"


def test_main_runs_selected_action_and_writes_report(monkeypatch: object, tmp_path: Path) -> None:
    module = _load_script_module()

    conversations_path = tmp_path / "conversations.jsonl"
    rows = [
        {
            "conversation_id": "conv-1",
            "query": "How to reset password?",
            "expected_answer": "Use admin console.",
            "answer": "Use admin console.",
        }
    ]
    conversations_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    args = [
        "run_observability.py",
        "continuous-eval",
        "--tracking-uri",
        f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}",
        "--reports-dir",
        str(tmp_path / "reports"),
        "--conversations-path",
        str(conversations_path),
        "--continuous-sample-ratio",
        "1.0",
    ]
    monkeypatch.setattr(sys, "argv", args)

    module.main()

    report_path = tmp_path / "reports" / "observability_run_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "continuous-eval"
    assert payload["continuous_evaluation"]["sample_size"] == 1
