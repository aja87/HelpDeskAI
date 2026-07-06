"""LLMOps, observability and FinOps helpers for HelpDeskAI."""

from helpdeskai.observability.continuous_eval import (
    detect_drift,
    judge_answer,
    load_conversation_samples,
)
from helpdeskai.observability.finops import Scenario, make_current_poc, make_optimized, recommend
from helpdeskai.observability.mlflow_model import (
    DEFAULT_RAG_MODEL_EXPERIMENT,
    DEFAULT_RAG_MODEL_NAME,
    RagChainPyfuncModel,
    register_rag_pyfunc_model,
)
from helpdeskai.observability.mlflow_tracking import (
    configure_mlflow,
    estimate_rag_cost_usd,
    log_rag_evaluation_run,
)
from helpdeskai.observability.prompt_registry import (
    load_prompt_by_alias,
    promote_prompt_alias,
    register_prompt_versions,
)

__all__ = [
    "Scenario",
    "DEFAULT_RAG_MODEL_EXPERIMENT",
    "DEFAULT_RAG_MODEL_NAME",
    "RagChainPyfuncModel",
    "configure_mlflow",
    "detect_drift",
    "estimate_rag_cost_usd",
    "judge_answer",
    "load_conversation_samples",
    "load_prompt_by_alias",
    "log_rag_evaluation_run",
    "make_current_poc",
    "make_optimized",
    "promote_prompt_alias",
    "recommend",
    "register_rag_pyfunc_model",
    "register_prompt_versions",
]
