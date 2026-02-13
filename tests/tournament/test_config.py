from pathlib import Path

import pytest
from pydantic import ValidationError

from inspect_ai.tournament import TournamentConfig, load_tournament_config


def test_load_tournament_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "tournament.yaml"
    config_path.write_text(
        """
completion_log_dir: ./logs/completions
log_dir: ./logs/tournament
contestant_models:
  - model/a
  - model/b
prompts:
  - id: p-1
    text: Prompt one
judge_model: judge/model
judge_prompt_template: |
  Prompt:
  {prompt}
  A:
  {response_a}
  B:
  {response_b}
""".strip(),
        encoding="utf-8",
    )

    config = load_tournament_config(config_path)
    expected_root = config_path.parent.resolve()
    assert config.completion_log_dir == expected_root / "logs" / "completions"
    assert config.log_dir == expected_root / "logs" / "tournament"
    assert config.state_dir == expected_root / "logs" / "tournament" / "state"
    assert config.invalid_policy == "skip"


def test_tournament_config_rejects_duplicate_prompts() -> None:
    with pytest.raises(ValidationError):
        TournamentConfig.model_validate(
            {
                "completion_log_dir": "logs/completions",
                "log_dir": "logs/tournament",
                "contestant_models": ["model/a", "model/b"],
                "prompts": [
                    {"id": "prompt-1", "text": "Prompt one"},
                    {"id": "prompt-1", "text": "Prompt duplicate"},
                ],
                "judge_model": "judge/model",
                "judge_prompt_template": "Prompt: {prompt}",
            }
        )


def test_tournament_config_rejects_invalid_pair_bounds() -> None:
    with pytest.raises(ValidationError):
        TournamentConfig.model_validate(
            {
                "completion_log_dir": "logs/completions",
                "log_dir": "logs/tournament",
                "contestant_models": ["model/a", "model/b"],
                "prompts": [{"id": "prompt-1", "text": "Prompt one"}],
                "judge_model": "judge/model",
                "judge_prompt_template": "Prompt: {prompt}",
                "min_pair_matches": 5,
                "max_pair_matches": 2,
            }
        )
