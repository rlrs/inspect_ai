from inspect_ai.tournament import TournamentConfig
from inspect_ai.tournament.stopping import (
    check_convergence,
    check_hard_stops,
    probability_higher,
)
from inspect_ai.tournament.types import ModelRating


def test_probability_higher_behaves_monotonically() -> None:
    high = probability_higher(
        mu_higher=30.0,
        sigma_higher=2.0,
        mu_lower=20.0,
        sigma_lower=2.0,
        beta=4.1666666667,
    )
    low = probability_higher(
        mu_higher=22.0,
        sigma_higher=8.0,
        mu_lower=21.0,
        sigma_lower=8.0,
        beta=4.1666666667,
    )
    assert high > low
    assert 0.5 < high < 1.0
    assert 0.0 < low < 1.0


def test_convergence_requires_stable_batches() -> None:
    config = build_rating_config()
    ratings = {
        "a": ModelRating(model_id="a", mu=45.0, sigma=1.0),
        "b": ModelRating(model_id="b", mu=25.0, sigma=1.0),
        "c": ModelRating(model_id="c", mu=5.0, sigma=1.0),
    }
    first = check_convergence(
        ratings,
        rating_params=config.rating_params,
        p_stop=0.98,
        epsilon=0.15,
        conservative_k=config.conservative_k,
        n_stable_batches=2,
        stable_batches=0,
        elo_scale=config.elo_scale,
    )
    second = check_convergence(
        ratings,
        rating_params=config.rating_params,
        p_stop=0.98,
        epsilon=0.15,
        conservative_k=config.conservative_k,
        n_stable_batches=2,
        stable_batches=first.stable_batches,
        elo_scale=config.elo_scale,
    )
    assert first.satisfied_now is True
    assert first.converged is False
    assert second.satisfied_now is True
    assert second.converged is True


def test_convergence_resets_stability_when_uncertain() -> None:
    config = build_rating_config()
    uncertain = {
        "a": ModelRating(model_id="a", mu=25.5, sigma=8.0),
        "b": ModelRating(model_id="b", mu=25.0, sigma=8.0),
        "c": ModelRating(model_id="c", mu=24.5, sigma=8.0),
    }
    result = check_convergence(
        uncertain,
        rating_params=config.rating_params,
        p_stop=0.98,
        epsilon=0.15,
        conservative_k=config.conservative_k,
        n_stable_batches=3,
        stable_batches=2,
        elo_scale=config.elo_scale,
    )
    assert result.satisfied_now is False
    assert result.stable_batches == 0
    assert result.converged is False


def test_hard_stop_conditions() -> None:
    maxed = check_hard_stops(
        total_matches=100,
        max_total_matches=100,
        no_eligible_pairs=False,
    )
    exhausted = check_hard_stops(
        total_matches=10,
        max_total_matches=100,
        no_eligible_pairs=True,
    )
    assert maxed.should_stop is True
    assert "max_total_matches_reached" in maxed.reasons
    assert exhausted.should_stop is True
    assert "no_eligible_pairs" in exhausted.reasons


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
