import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from pydantic import BaseModel, Field

from inspect_ai import Task, eval_set
from inspect_ai._util._async import run_coroutine
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, get_model

from ._trace import tournament_trace_file
from .config import TournamentConfig, load_tournament_config
from .types import default_project_id

TOURNAMENT_PHASE_KEY = "inspect_ai:tournament_phase"
TOURNAMENT_PROJECT_KEY = "inspect_ai:tournament_project_id"
GENERATION_PHASE = "generation"

logger = logging.getLogger(__name__)


class GenerationRunResult(BaseModel):
    """Result summary for a generation run."""

    models: list[str]
    prompt_count: int
    log_dir: Path
    log_count: int
    logs: list[EvalLog] = Field(default_factory=list)


def build_generation_task(config: TournamentConfig) -> Task:
    """Create generation task from tournament prompts."""
    samples: list[Sample] = [
        Sample(
            id=prompt.id,
            input=prompt.text,
            metadata={config.prompt_id_field: prompt.id},
        )
        for prompt in config.prompts
    ]
    generate_config = GenerateConfig.model_validate(config.contestant_generate_config)
    return Task(
        name="tournament_generation",
        dataset=samples,
        config=generate_config,
    )


def run_generation(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    models: Sequence[str] | None = None,
) -> GenerationRunResult:
    """Run model generation for tournament prompts."""
    parsed = load_tournament_config(config)
    selected_models = list(models) if models is not None else parsed.contestant_models
    if len(selected_models) == 0:
        raise ValueError("At least one model is required for generation")

    unknown_models = sorted(set(selected_models) - set(parsed.contestant_models))
    if len(unknown_models) > 0:
        raise ValueError(
            "Unknown model(s) requested for generation: "
            + ", ".join(unknown_models)
            + "."
        )

    task = build_generation_task(parsed)
    project_id = parsed.project_id or default_project_id(
        parsed.contestant_models,
        [prompt.id for prompt in parsed.prompts],
        seed=parsed.seed,
    )
    metadata = {
        TOURNAMENT_PHASE_KEY: GENERATION_PHASE,
        TOURNAMENT_PROJECT_KEY: project_id,
    }
    model_args = _resolve_env_model_args()
    logs: list[EvalLog] = []
    with tournament_trace_file(parsed.log_dir, GENERATION_PHASE):
        for model_name in selected_models:
            model = get_model(model_name, **(model_args | {"memoize": False}))
            try:
                success, model_logs = eval_set(
                    tasks=task,
                    model=model,
                    log_dir=parsed.completion_log_dir.as_posix(),
                    metadata=metadata,
                    score=False,
                    log_dir_allow_dirty=True,
                )
                logs.extend(model_logs)
                if not success:
                    raise RuntimeError(
                        "Generation run did not complete successfully for model "
                        + f"'{model_name}'"
                    )
            finally:
                _close_model(model)

    return GenerationRunResult(
        models=selected_models,
        prompt_count=len(parsed.prompts),
        log_dir=parsed.completion_log_dir,
        log_count=len(logs),
        logs=logs,
    )


def _close_model(model: Model) -> None:
    """Close model resources created for generation, including local servers."""
    try:
        model.__exit__(None, None, None)
        return
    except RuntimeError as ex:
        if "require an async close" not in str(ex):
            logger.warning(f"Error while closing model '{model}': {ex}")
            return
    except Exception as ex:
        logger.warning(f"Error while closing model '{model}': {ex}")
        return

    try:
        run_coroutine(model.__aexit__(None, None, None))
    except Exception as ex:
        logger.warning(f"Error while closing model '{model}': {ex}")


def _resolve_env_model_args() -> dict[str, Any]:
    model_args: dict[str, Any] = {}
    env_model_args = os.environ.get("INSPECT_EVAL_MODEL_ARGS")
    if not env_model_args:
        return model_args

    for arg in env_model_args.split(" "):
        arg = arg.strip()
        if not arg or "=" not in arg:
            continue

        key, value_raw = arg.split("=", maxsplit=1)
        value = yaml.safe_load(value_raw)
        if isinstance(value, str):
            value_parts = value.split(",")
            value = value_parts if len(value_parts) > 1 else value_parts[0]
        model_args[key.replace("-", "_")] = value

    return model_args
