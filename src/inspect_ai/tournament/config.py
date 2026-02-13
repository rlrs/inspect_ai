from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field, JsonValue, model_validator
from typing_extensions import Self

from inspect_ai._util.config import read_config_object

from .types import InvalidPolicy


class TournamentPrompt(BaseModel):
    """Prompt used in the tournament."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, JsonValue] | None = Field(default=None)


class TrueSkillRatingParams(BaseModel):
    """TrueSkill hyperparameters."""

    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    beta: float = 25.0 / 6.0
    tau: float = 25.0 / 300.0
    draw_probability: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_positive(self) -> Self:
        if self.sigma <= 0.0:
            raise ValueError("rating_params.sigma must be greater than 0")
        if self.beta <= 0.0:
            raise ValueError("rating_params.beta must be greater than 0")
        if self.tau <= 0.0:
            raise ValueError("rating_params.tau must be greater than 0")
        return self


class TournamentConfig(BaseModel):
    """Tournament configuration schema."""

    completion_log_dir: Path
    log_dir: Path
    state_dir: Path | None = Field(default=None)
    project_id: str | None = Field(default=None, min_length=1)

    contestant_models: list[str] = Field(min_length=2)
    prompts: list[TournamentPrompt] = Field(min_length=1)
    regenerate_completions: bool = False
    contestant_generate_config: dict[str, JsonValue] = Field(default_factory=dict)

    rating_engine: str = "trueskill"
    rating_params: TrueSkillRatingParams = Field(default_factory=TrueSkillRatingParams)
    conservative_k: float = Field(default=3.0, gt=0.0)
    elo_scale: float = Field(default=173.7178, gt=0.0)

    batch_size: int = Field(default=8, gt=0)
    max_total_matches: int = Field(default=128, gt=0)
    p_stop: float = Field(default=0.98, gt=0.0, le=1.0)
    epsilon: float = Field(default=0.15, ge=0.0)
    min_pair_matches: int = Field(default=2, ge=0)
    max_pair_matches: int = Field(default=24, gt=0)
    max_prompt_uses_per_pair: int = Field(default=3, gt=0)
    top_k_focus: int = Field(default=5, gt=0)
    n_stable_batches: int = Field(default=3, gt=0)
    seed: int = 42

    judge_model: str = Field(min_length=1)
    judge_max_samples: int = Field(default=8, gt=0)
    judge_max_concurrency: int = Field(default=64, gt=0)
    judge_generate_config: dict[str, JsonValue] = Field(default_factory=dict)
    judge_prompt_template: str = Field(min_length=1)
    side_swap: bool = True
    prompt_id_field: str = Field(default="prompt_id", min_length=1)

    invalid_policy: InvalidPolicy = "skip"
    bootstrap_matches_per_new_model: int = Field(default=12, gt=0)

    model_config = {
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if self.rating_engine != "trueskill":
            raise ValueError("rating_engine must be 'trueskill'")

        if self.min_pair_matches > self.max_pair_matches:
            raise ValueError("min_pair_matches cannot exceed max_pair_matches")

        duplicate_models = _duplicates(self.contestant_models)
        if len(duplicate_models) > 0:
            raise ValueError(
                f"contestant_models contains duplicates: {', '.join(duplicate_models)}"
            )

        prompt_ids = [prompt.id for prompt in self.prompts]
        duplicate_prompt_ids = _duplicates(prompt_ids)
        if len(duplicate_prompt_ids) > 0:
            raise ValueError(
                f"prompts contains duplicate ids: {', '.join(duplicate_prompt_ids)}"
            )

        if self.state_dir is None:
            self.state_dir = self.log_dir / "state"

        return self

    def with_resolved_paths(self, base_dir: Path) -> "TournamentConfig":
        """Resolve all path fields against a base directory."""
        resolved_log_dir = _resolve_path(base_dir, self.log_dir)
        resolved_completion_log_dir = _resolve_path(base_dir, self.completion_log_dir)
        resolved_state_dir = _resolve_path(base_dir, self.state_dir or self.log_dir)
        return self.model_copy(
            update={
                "log_dir": resolved_log_dir,
                "completion_log_dir": resolved_completion_log_dir,
                "state_dir": resolved_state_dir,
            }
        )


def load_tournament_config(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    """Load and validate a tournament config from object or file."""
    if isinstance(config, TournamentConfig):
        return config

    if isinstance(config, Mapping):
        return TournamentConfig.model_validate(dict(config))

    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Tournament config file not found: {config_path}")

    config_text = config_path.read_text(encoding="utf-8")
    config_dict = read_config_object(config_text)
    parsed = TournamentConfig.model_validate(config_dict)
    return parsed.with_resolved_paths(config_path.parent)


def _resolve_path(base_dir: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base_dir / value).resolve()


def _duplicates(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted([value for value, count in counts.items() if count > 1])
