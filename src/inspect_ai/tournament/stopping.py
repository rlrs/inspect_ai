import math
from collections.abc import Mapping

from pydantic import BaseModel

from .config import TournamentConfig, TrueSkillRatingParams
from .rating import summarize_ratings
from .types import ModelRating


class AdjacentCheck(BaseModel):
    """Convergence check for one adjacent ranking pair."""

    higher_model: str
    lower_model: str
    probability_higher: float
    conservative_margin: float
    passes_probability: bool
    passes_margin: bool


class ConvergenceCheck(BaseModel):
    """Batch-level convergence status."""

    satisfied_now: bool
    converged: bool
    stable_batches: int
    adjacent: list[AdjacentCheck]

    @property
    def min_probability(self) -> float | None:
        """Minimum adjacent win probability."""
        if len(self.adjacent) == 0:
            return None
        return min(check.probability_higher for check in self.adjacent)

    @property
    def min_margin(self) -> float | None:
        """Minimum adjacent conservative margin."""
        if len(self.adjacent) == 0:
            return None
        return min(check.conservative_margin for check in self.adjacent)


class HardStopCheck(BaseModel):
    """Hard stop decision for scheduling loop."""

    should_stop: bool
    reasons: list[str]


def check_convergence(
    ratings: Mapping[str, ModelRating],
    *,
    rating_params: TrueSkillRatingParams,
    p_stop: float,
    epsilon: float,
    conservative_k: float,
    n_stable_batches: int,
    stable_batches: int,
    elo_scale: float = 173.7178,
) -> ConvergenceCheck:
    """Evaluate soft convergence for adjacent conservative ranks."""
    standings = summarize_ratings(
        ratings,
        params=rating_params,
        conservative_k=conservative_k,
        elo_scale=elo_scale,
    )
    if len(standings) <= 1:
        next_stable = stable_batches + 1
        return ConvergenceCheck(
            satisfied_now=True,
            converged=next_stable >= n_stable_batches,
            stable_batches=next_stable,
            adjacent=[],
        )

    adjacent_checks: list[AdjacentCheck] = []
    for index in range(0, len(standings) - 1):
        higher = standings[index]
        lower = standings[index + 1]
        probability = probability_higher(
            mu_higher=higher.mu,
            sigma_higher=higher.sigma,
            mu_lower=lower.mu,
            sigma_lower=lower.sigma,
            beta=rating_params.beta,
        )
        margin = higher.conservative - lower.conservative
        adjacent_checks.append(
            AdjacentCheck(
                higher_model=higher.model_id,
                lower_model=lower.model_id,
                probability_higher=probability,
                conservative_margin=margin,
                passes_probability=probability >= p_stop,
                passes_margin=margin >= epsilon,
            )
        )

    satisfied_now = all(
        check.passes_probability and check.passes_margin for check in adjacent_checks
    )
    next_stable = (stable_batches + 1) if satisfied_now else 0
    return ConvergenceCheck(
        satisfied_now=satisfied_now,
        converged=next_stable >= n_stable_batches,
        stable_batches=next_stable,
        adjacent=adjacent_checks,
    )


def probability_higher(
    *,
    mu_higher: float,
    sigma_higher: float,
    mu_lower: float,
    sigma_lower: float,
    beta: float,
) -> float:
    """Compute probability that higher-ranked model beats lower-ranked model."""
    denom = math.sqrt((2.0 * (beta**2)) + (sigma_higher**2) + (sigma_lower**2))
    if denom == 0.0:
        return 0.5
    return _normal_cdf((mu_higher - mu_lower) / denom)


def check_hard_stops(
    *,
    total_matches: int,
    max_total_matches: int,
    no_eligible_pairs: bool,
) -> HardStopCheck:
    """Evaluate hard stop conditions."""
    reasons: list[str] = []
    if total_matches >= max_total_matches:
        reasons.append("max_total_matches_reached")
    if no_eligible_pairs:
        reasons.append("no_eligible_pairs")
    return HardStopCheck(should_stop=len(reasons) > 0, reasons=reasons)


def check_hard_stops_for_config(
    config: TournamentConfig,
    *,
    total_matches: int,
    no_eligible_pairs: bool,
) -> HardStopCheck:
    """Evaluate hard stop conditions from config values."""
    return check_hard_stops(
        total_matches=total_matches,
        max_total_matches=config.max_total_matches,
        no_eligible_pairs=no_eligible_pairs,
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
