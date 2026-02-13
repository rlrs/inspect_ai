from pathlib import Path

from inspect_ai.model import get_model
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.tournament import TournamentConfig
from inspect_ai.tournament.judge_task import (
    JudgeMatch,
    build_judge_samples,
    run_judge_batch,
)
from inspect_ai.tournament.scorer import reconcile_side_swap


def test_build_judge_samples_side_swap_shapes_dataset() -> None:
    match = JudgeMatch(
        match_id="m-1",
        prompt_id="p-1",
        prompt="Prompt text",
        model_a="model/a",
        model_b="model/b",
        response_a="Response A",
        response_b="Response B",
    )
    samples = build_judge_samples([match], side_swap=True)
    assert len(samples) == 2
    assert samples[0].id == "m-1:ab"
    assert samples[1].id == "m-1:ba"
    assert samples[0].metadata and samples[0].metadata["side"] == "ab"
    assert samples[1].metadata and samples[1].metadata["side"] == "ba"


def test_run_judge_batch_parses_side_swapped_decisions_deterministically(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    match = JudgeMatch(
        match_id="m-1",
        prompt_id="p-1",
        prompt="Write one sentence.",
        model_a="model/a",
        model_b="model/b",
        response_a="A response",
        response_b="B response",
    )
    grader_model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", "Reasoning.\nDECISION: A"),
            ModelOutput.from_content("mockllm/model", "Reasoning.\nDECISION: B"),
        ],
    )

    result = run_judge_batch(config, [match], grader_model=grader_model)
    assert result.log_count == 1
    assert result.sample_count == 2
    assert len(result.logs) == 1
    assert result.logs[0].samples is not None

    samples = result.logs[0].samples
    score_ab = next(
        sample.scores["pairwise_judge"]
        for sample in samples
        if sample.id == "m-1:ab"
    )
    score_ba = next(
        sample.scores["pairwise_judge"]
        for sample in samples
        if sample.id == "m-1:ba"
    )
    assert score_ab.value == "A"
    assert score_ba.value == "B"
    assert reconcile_side_swap(score_ab.value, score_ba.value) == "A"


def build_config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": tmp_path / "logs" / "completions",
            "log_dir": tmp_path / "logs" / "tournament",
            "state_dir": tmp_path / "logs" / "tournament" / "state",
            "contestant_models": ["model/a", "model/b"],
            "prompts": [{"id": "p-1", "text": "Prompt one"}],
            "judge_model": "mockllm/model",
            "judge_max_samples": 4,
            "judge_generate_config": {"max_tokens": 64},
            "judge_prompt_template": (
                "Prompt:\n{prompt}\nA:\n{response_a}\nB:\n{response_b}\n"
                "DECISION only at end."
            ),
            "side_swap": True,
        }
    )
