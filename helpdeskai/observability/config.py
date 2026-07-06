from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


REPORTS_DIR = Path("reports/observability")
TRACES_DIR = REPORTS_DIR / "traces"
LOG_FILE = "observability.log"

DEFAULT_GOLDEN_PATH = Path("tests/golden/golden_dataset.jsonl")
DEFAULT_CONVERSATIONS_PATH = REPORTS_DIR / "simulated_conversations.jsonl"

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(REPORTS_DIR / 'mlflow.db').resolve()}")

VALID_ACTIONS = {
    "track",
    "prompts",
    "model",
    "trace",
    "finops",
    "continuous-eval",
    "all",
}


@dataclass(slots=True)
class ObservabilityConfig:
    """Runtime configuration for observability pipelines."""

    reports_dir: Path = REPORTS_DIR
    traces_dir: Path = TRACES_DIR
    golden_path: Path = DEFAULT_GOLDEN_PATH
    conversations_path: Path = DEFAULT_CONVERSATIONS_PATH

    tracking_uri: str = DEFAULT_TRACKING_URI
    eval_experiment: str = "observability-eval"
    prompt_experiment: str = "observability-prompts"
    model_experiment: str = "observability-models"
    continuous_eval_experiment: str = "observability-continuous-eval"

    prompt_name: str = "rag-system"
    retrieval_mode: str = "hybrid"
    generator_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "claude-sonnet-4-6"
    prompt_registry_dev_version: str = "v1"
    prompt_registry_staging_version: str = "v2"

    registered_model_name: str = "helpdeskai-rag-pyfunc"
    production_alias: str = "production"

    monthly_budget_usd: float = 100.0
    continuous_sample_ratio: float = 0.05
    seed: int = 42

    def validate(self) -> None:
        """Validate observability runtime parameters."""

        if not self.tracking_uri.strip():
            raise ValueError("tracking_uri must not be empty")
        if self.monthly_budget_usd <= 0:
            raise ValueError("monthly_budget_usd must be > 0")
        if not 0 < self.continuous_sample_ratio <= 1:
            raise ValueError("continuous_sample_ratio must be in (0, 1]")
        if not self.prompt_name.strip():
            raise ValueError("prompt_name must not be empty")
        if not self.generator_model.strip():
            raise ValueError("generator_model must not be empty")
        if not self.judge_model.strip():
            raise ValueError("judge_model must not be empty")
        if not self.registered_model_name.strip():
            raise ValueError("registered_model_name must not be empty")
        if not self.production_alias.strip():
            raise ValueError("production_alias must not be empty")
