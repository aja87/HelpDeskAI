"""Prompt registry pattern implemented with MLflow runs and aliases."""

from __future__ import annotations

from collections.abc import Mapping

PROMPT_NAME = "rag-system-prompt"
PROMPT_EXPERIMENT = "helpdeskai-prompts"


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def _experiment_id(experiment_name: str) -> str:
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        return str(experiment_id)
    return str(experiment.experiment_id)


def register_prompt_versions(
    prompts: dict[str, str],
    aliases: dict[str, str],
    *,
    prompt_name: str = PROMPT_NAME,
    experiment: str = PROMPT_EXPERIMENT,
) -> dict[str, str]:
    """Register prompt texts as MLflow runs and assign aliases to versions."""
    import mlflow

    mlflow.set_experiment(experiment)
    run_ids: dict[str, str] = {}
    for version, prompt_text in prompts.items():
        with mlflow.start_run(run_name=f"register_{prompt_name}_{version}") as run:
            mlflow.log_text(prompt_text, f"{prompt_name}.txt")
            mlflow.log_params(
                {
                    "prompt_name": prompt_name,
                    "version_label": version,
                    "template_length": len(prompt_text),
                }
            )
            run_ids[version] = run.info.run_id

    client = _client()
    for alias, version in aliases.items():
        if version not in run_ids:
            raise ValueError(f"alias '{alias}' points to unknown prompt version '{version}'")
        promote_prompt_alias(
            prompt_name,
            version,
            alias,
            experiment=experiment,
            run_ids=run_ids,
            client=client,
        )
    return run_ids


def load_prompt_by_alias(
    prompt_name: str,
    alias: str,
    *,
    experiment: str = PROMPT_EXPERIMENT,
) -> tuple[str, str] | None:
    """Load a prompt version and text by alias."""
    client = _client()
    runs = client.search_runs(
        experiment_ids=[_experiment_id(experiment)],
        filter_string=(
            f"tags.`alias.{alias}` = 'current' AND params.prompt_name = '{prompt_name}'"
        ),
        max_results=1,
    )
    if not runs:
        return None
    run = runs[0]
    local_path = client.download_artifacts(run.info.run_id, f"{prompt_name}.txt")
    with open(local_path, encoding="utf-8") as handle:
        return run.data.params.get("version_label", ""), handle.read()


def promote_prompt_alias(
    prompt_name: str,
    version: str,
    alias: str,
    *,
    experiment: str = PROMPT_EXPERIMENT,
    run_ids: Mapping[str, str] | None = None,
    client=None,
) -> None:
    """Move an alias to a given prompt version."""
    client = client or _client()
    experiment_id = _experiment_id(experiment)
    current_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`alias.{alias}` = 'current' AND params.prompt_name = '{prompt_name}'",
    )
    for run in current_runs:
        client.delete_tag(run.info.run_id, f"alias.{alias}")

    target_run_id = None
    if run_ids and version in run_ids:
        target_run_id = run_ids[version]
    else:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"params.prompt_name = '{prompt_name}' AND params.version_label = '{version}'"
            ),
            max_results=1,
        )
        if runs:
            target_run_id = runs[0].info.run_id
    if target_run_id is None:
        raise ValueError(f"prompt version '{version}' not found for '{prompt_name}'")
    client.set_tag(target_run_id, f"alias.{alias}", "current")
