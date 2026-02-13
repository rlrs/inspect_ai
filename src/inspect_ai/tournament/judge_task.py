from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, JsonValue

from inspect_ai import Epochs, Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.solver._solver import Solver

from .config import TournamentConfig, load_tournament_config
from .scorer import pairwise_judge


class JudgeMatch(BaseModel):
    """Pairwise match to be judged."""

    match_id: str
    prompt_id: str
    prompt: str
    model_a: str
    model_b: str
    model_a_id: str | None = None
    model_b_id: str | None = None
    response_a: str
    response_b: str
    metadata: dict[str, JsonValue] | None = None


class JudgeRunResult(BaseModel):
    """Result summary for a judge batch run."""

    match_count: int
    sample_count: int
    log_dir: Path
    log_count: int
    logs: list[EvalLog] = Field(default_factory=list)


@solver
def judge_noop_solver() -> Solver:
    """No-op solver used for judge-only tasks."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        return state

    return solve


def build_judge_samples(
    matches: Sequence[JudgeMatch],
    *,
    side_swap: bool = True,
) -> list[Sample]:
    """Build judge task samples from scheduled matches."""
    samples: list[Sample] = []
    for match in matches:
        base_metadata: dict[str, JsonValue] = {
            "match_id": match.match_id,
            "prompt_id": match.prompt_id,
            "prompt": match.prompt,
        }
        if match.metadata is not None:
            base_metadata.update(match.metadata)

        ab_metadata: dict[str, JsonValue] = {
            **base_metadata,
            "side": "ab",
            "model_a": match.model_a,
            "model_b": match.model_b,
            "response_a": match.response_a,
            "response_b": match.response_b,
        }
        if match.model_a_id is not None:
            ab_metadata["model_a_id"] = match.model_a_id
        if match.model_b_id is not None:
            ab_metadata["model_b_id"] = match.model_b_id
        samples.append(
            Sample(
                id=f"{match.match_id}:ab" if side_swap else match.match_id,
                input=match.prompt,
                metadata=ab_metadata,
            )
        )

        if side_swap:
            ba_metadata: dict[str, JsonValue] = {
                **base_metadata,
                "side": "ba",
                "model_a": match.model_b,
                "model_b": match.model_a,
                "response_a": match.response_b,
                "response_b": match.response_a,
            }
            if match.model_b_id is not None:
                ba_metadata["model_a_id"] = match.model_b_id
            if match.model_a_id is not None:
                ba_metadata["model_b_id"] = match.model_a_id
            samples.append(
                Sample(
                    id=f"{match.match_id}:ba",
                    input=match.prompt,
                    metadata=ba_metadata,
                )
            )

    return samples


def build_judge_task(
    config: TournamentConfig,
    matches: Sequence[JudgeMatch],
) -> Task:
    """Build a task that judges pairwise match samples."""
    return Task(
        name="tournament_judge",
        dataset=build_judge_samples(matches, side_swap=config.side_swap),
        solver=judge_noop_solver(),
        scorer=pairwise_judge(
            prompt_template=config.judge_prompt_template,
            model_role="grader",
            generate_config=config.judge_generate_config,
        ),
    )


def run_judge_batch(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    matches: Sequence[JudgeMatch],
    *,
    grader_model: str | Model | None = None,
    log_dir: str | Path | None = None,
) -> JudgeRunResult:
    """Run one judge batch and return created eval logs."""
    parsed = load_tournament_config(config)
    task = build_judge_task(parsed, matches)
    judge_log_dir = (
        Path(log_dir)
        if log_dir is not None
        else parsed.log_dir / "judge"
    )

    grader = _resolve_grader_model(parsed, grader_model)
    logs = eval(
        tasks=task,
        model=parsed.judge_model,
        model_roles={"grader": grader},
        log_dir=judge_log_dir.as_posix(),
        epochs=Epochs(1, []),
        max_samples=parsed.judge_max_samples,
    )
    failed_logs = [log for log in logs if log.status != "success"]
    if len(failed_logs) > 0:
        raise RuntimeError(
            f"Judge batch did not complete successfully ({len(failed_logs)} failed logs)."
        )

    return JudgeRunResult(
        match_count=len(matches),
        sample_count=len(build_judge_samples(matches, side_swap=parsed.side_swap)),
        log_dir=judge_log_dir,
        log_count=len(logs),
        logs=logs,
    )


def _resolve_grader_model(
    config: TournamentConfig,
    grader_model: str | Model | None,
) -> Model:
    if isinstance(grader_model, Model):
        return grader_model

    model_name = config.judge_model if grader_model is None else grader_model
    generate_config = GenerateConfig.model_validate(config.judge_generate_config)
    return get_model(model_name, config=generate_config)
