from pathlib import Path

from inspect_ai.tournament import TournamentConfig, TournamentStore
from inspect_ai.tournament.rating import (
    MatchOutcome,
    apply_outcomes,
    apply_outcomes_to_store,
)
from inspect_ai.tournament.store import initialize_tournament_store
from inspect_ai.tournament.types import ModelRating


def test_apply_outcomes_tracks_wins_losses_ties_and_invalid_skip() -> None:
    ratings = {
        "a": ModelRating(model_id="a", mu=25.0, sigma=8.3333333333),
        "b": ModelRating(model_id="b", mu=25.0, sigma=8.3333333333),
    }
    outcomes = [
        MatchOutcome(model_a="a", model_b="b", decision="A"),
        MatchOutcome(model_a="a", model_b="b", decision="TIE"),
        MatchOutcome(model_a="a", model_b="b", decision="INVALID"),
    ]
    result = apply_outcomes(
        ratings,
        outcomes,
        params=build_rating_config().rating_params,
        invalid_policy="skip",
    )

    assert result.processed_outcomes == 2
    assert result.skipped_outcomes == 1
    assert result.ratings["a"].games == 2
    assert result.ratings["b"].games == 2
    assert result.ratings["a"].wins == 1
    assert result.ratings["b"].losses == 1
    assert result.ratings["a"].ties == 1
    assert result.ratings["b"].ties == 1
    assert result.ratings["a"].mu > 25.0
    assert result.ratings["b"].mu < 25.0


def test_apply_outcomes_invalid_as_tie() -> None:
    ratings = {
        "a": ModelRating(model_id="a", mu=25.0, sigma=8.3333333333),
        "b": ModelRating(model_id="b", mu=25.0, sigma=8.3333333333),
    }
    result = apply_outcomes(
        ratings,
        [MatchOutcome(model_a="a", model_b="b", decision="INVALID")],
        params=build_rating_config().rating_params,
        invalid_policy="count_as_tie",
    )
    assert result.processed_outcomes == 1
    assert result.skipped_outcomes == 0
    assert result.ratings["a"].games == 1
    assert result.ratings["a"].ties == 1
    assert result.ratings["b"].games == 1
    assert result.ratings["b"].ties == 1


def test_apply_outcomes_to_store_persists_ratings_and_history(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    initialize_tournament_store(config)

    with TournamentStore(config.state_dir) as store:
        model_a = store.model_identifier("model/a")
        model_b = store.model_identifier("model/b")
        assert model_a is not None
        assert model_b is not None

        first = apply_outcomes_to_store(
            store,
            config,
            [MatchOutcome(model_a=model_a, model_b=model_b, decision="A")],
        )
        second = apply_outcomes_to_store(
            store,
            config,
            [MatchOutcome(model_a=model_a, model_b=model_b, decision="TIE")],
        )

        assert first.step_id == 1
        assert second.step_id == 2
        assert store.table_count("ratings_history") == 4

        ratings = store.load_model_ratings()
        assert ratings[model_a].games == 2
        assert ratings[model_b].games == 2


def build_rating_config() -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": "logs/completions",
            "log_dir": "logs/tournament",
            "contestant_models": ["model/a", "model/b"],
            "prompts": [{"id": "p-1", "text": "Prompt"}],
            "judge_model": "judge/model",
            "judge_prompt_template": "Prompt: {prompt}",
        }
    )


def build_config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": tmp_path / "logs" / "completions",
            "log_dir": tmp_path / "logs" / "tournament",
            "state_dir": tmp_path / "logs" / "tournament" / "state",
            "contestant_models": ["model/a", "model/b"],
            "prompts": [{"id": "p-1", "text": "Prompt"}],
            "judge_model": "judge/model",
            "judge_prompt_template": "Prompt: {prompt}",
        }
    )
