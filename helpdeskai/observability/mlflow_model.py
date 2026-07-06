"""MLflow pyfunc packaging for the complete RAG chain."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.pyfunc import PythonModel

from helpdeskai.observability.mlflow_tracking import configure_mlflow
from helpdeskai.rag.models import RagConfig, RagResult
from helpdeskai.rag.pipeline import AdvancedRagPipeline

DEFAULT_RAG_MODEL_EXPERIMENT = "helpdeskai-rag-models"
DEFAULT_RAG_MODEL_NAME = "helpdeskai-rag-chain"

PipelineFactory = Callable[[RagConfig], AdvancedRagPipeline]


def _config_dict(config: RagConfig) -> dict[str, Any]:
    return asdict(config)


def _config_from_mapping(values: dict[str, Any] | RagConfig | None) -> RagConfig:
    if values is None:
        return RagConfig()
    if isinstance(values, RagConfig):
        return values
    return RagConfig(**values)


def _coerce_questions(model_input: Any) -> list[str]:
    if isinstance(model_input, pd.DataFrame):
        if "question" not in model_input.columns:
            raise ValueError("RAG pyfunc input DataFrame must contain a 'question' column")
        return [str(value) for value in model_input["question"].tolist()]
    if isinstance(model_input, pd.Series):
        return [str(value) for value in model_input.tolist()]
    if isinstance(model_input, dict):
        if "question" in model_input:
            value = model_input["question"]
            if isinstance(value, str):
                return [value]
            if isinstance(value, Sequence):
                return [str(item) for item in value]
            return [str(value)]
        if "questions" in model_input:
            values = model_input["questions"]
            if isinstance(values, str):
                return [values]
            return [str(value) for value in values]
    if isinstance(model_input, str):
        return [model_input]
    if isinstance(model_input, Sequence):
        return [str(value) for value in model_input]
    raise ValueError("RAG pyfunc input must be a DataFrame, Series, dict, string, or sequence")


def _result_row(result: RagResult) -> dict[str, Any]:
    payload = result.to_dict()
    return {
        "question": payload["question_original"],
        "question_rewritten": payload["question_rewritten"],
        "answer": payload["answer"],
        "sources": payload["sources"],
        "contexts": payload["contexts"],
        "timings": payload["timings"],
        "model_names": payload["model_names"],
        "prompt_version": payload["prompt_version"],
        "retrieval_mode": payload["retrieval_mode"],
    }


class RagChainPyfuncModel(PythonModel):
    """MLflow-compatible wrapper around `AdvancedRagPipeline`."""

    def __init__(
        self,
        config: RagConfig | dict[str, Any] | None = None,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.config = _config_dict(_config_from_mapping(config))
        self._pipeline_factory = pipeline_factory
        self._pipeline: AdvancedRagPipeline | None = None

    def load_context(self, context: Any) -> None:
        """MLflow lifecycle hook; pipeline construction stays lazy."""
        self._pipeline = None

    def _get_pipeline(self) -> AdvancedRagPipeline:
        if self._pipeline is None:
            config = _config_from_mapping(self.config)
            if self._pipeline_factory is None:
                self._pipeline = AdvancedRagPipeline(config=config)
            else:
                self._pipeline = self._pipeline_factory(config)
        return self._pipeline

    def predict(self, context, model_input, params=None):
        """Run the complete RAG chain for one or more questions."""
        questions = _coerce_questions(model_input)
        results = [self._get_pipeline().run(question) for question in questions]
        return pd.DataFrame([_result_row(result) for result in results])


def register_rag_pyfunc_model(
    *,
    tracking_uri: str,
    experiment: str = DEFAULT_RAG_MODEL_EXPERIMENT,
    run_name: str = "register_rag_chain",
    model_name: str = DEFAULT_RAG_MODEL_NAME,
    artifact_path: str = "rag_chain",
    config: RagConfig = RagConfig(),
    code_paths: Sequence[Path | str] | None = None,
    alias: str | None = "production",
) -> dict[str, str | None]:
    """Log and register the complete RAG chain as an MLflow pyfunc model."""
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.tracking import MlflowClient
    from mlflow.types.schema import ColSpec, Schema

    configure_mlflow(tracking_uri, experiment)
    model = RagChainPyfuncModel(config)
    input_example = pd.DataFrame({"question": ["How do I configure SAML login?"]})
    signature = ModelSignature(inputs=Schema([ColSpec("string", "question")]))
    resolved_code_paths = [str(path) for path in code_paths] if code_paths else None

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_config_dict(config))
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            input_example=input_example,
            signature=signature,
            registered_model_name=model_name,
            code_paths=resolved_code_paths,
        )
        version = _find_registered_version(
            MlflowClient(),
            model_name=model_name,
            run_id=run.info.run_id,
        )
        if alias and version:
            MlflowClient().set_registered_model_alias(model_name, alias, version)
        return {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model_name": model_name,
            "model_version": version,
            "alias": alias if version else None,
        }


def _find_registered_version(
    client: Any,
    *,
    model_name: str,
    run_id: str,
) -> str | None:
    for version in client.search_model_versions(f"name = '{model_name}'"):
        if version.run_id == run_id:
            return str(version.version)
    return None
