import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel
from trueskill import Rating, TrueSkill, rate_1vs1  # type: ignore[import-untyped]

from .config import TournamentConfig, TrueSkillRatingParams
from .store import TournamentStore
from .types import Decision, InvalidPolicy, ModelRating


class MatchOutcome(BaseModel):
    """Canonical match outcome for rating updates."""

    model_a: str
    model_b: str
    decision: Decision


class ModelStanding(BaseModel):
    """Model rating plus derived ranking fields."""

    model_id: str
    model_name: str | None = None
    mu: float
    sigma: float
    games: int
    wins: int
    losses: int
    ties: int
    conservative: float
    elo_like: float


class RatingUpdateResult(BaseModel):
    """Result of applying a batch of match outcomes."""

    ratings: dict[str, ModelRating]
    standings: list[ModelStanding]
    processed_outcomes: int
    skipped_outcomes: int
    step_id: int | None = None


class TrueSkillEngine:
    """TrueSkill rating update helper."""

    def __init__(
        self,
        params: TrueSkillRatingParams,
        *,
        conservative_k: float = 3.0,
        elo_scale: float = 173.7178,
    ) -> None:
        self.params = params
        self.conservative_k = conservative_k
        self.elo_scale = elo_scale
        self.mu0 = params.mu
        self.env = TrueSkill(
            mu=params.mu,
            sigma=params.sigma,
            beta=params.beta,
            tau=params.tau,
            draw_probability=params.draw_probability,
        )

    def win_probability(self, first: ModelRating, second: ModelRating) -> float:
        """Probability that first beats second under current uncertainty."""
        denom = math.sqrt(
            (2.0 * (self.params.beta**2)) + (first.sigma**2) + (second.sigma**2)
        )
        if denom == 0.0:
            return 0.5
        return float(self.env.cdf((first.mu - second.mu) / denom))

    def conservative_score(self, rating: ModelRating) -> float:
        """Conservative score (mu - k*sigma)."""
        return rating.mu - (self.conservative_k * rating.sigma)

    def elo_like(self, rating: ModelRating) -> float:
        """Elo-like projection from TrueSkill mu."""
        return 1000.0 + self.elo_scale * (rating.mu - self.mu0)

    def apply_outcome(
        self,
        rating_a: ModelRating,
        rating_b: ModelRating,
        *,
        decision: Decision,
        invalid_policy: InvalidPolicy = "skip",
    ) -> tuple[ModelRating, ModelRating, bool, Decision]:
        """Apply one outcome to a pair of ratings."""
        if decision == "INVALID":
            if invalid_policy == "skip":
                return rating_a, rating_b, False, "INVALID"
            decision = "TIE"

        trueskill_a = Rating(mu=rating_a.mu, sigma=rating_a.sigma)
        trueskill_b = Rating(mu=rating_b.mu, sigma=rating_b.sigma)

        if decision == "A":
            updated_a, updated_b = rate_1vs1(
                trueskill_a,
                trueskill_b,
                drawn=False,
                env=self.env,
            )
        elif decision == "B":
            updated_b, updated_a = rate_1vs1(
                trueskill_b,
                trueskill_a,
                drawn=False,
                env=self.env,
            )
        else:
            updated_a, updated_b = rate_1vs1(
                trueskill_a,
                trueskill_b,
                drawn=True,
                env=self.env,
            )

        next_a = rating_a.model_copy(
            update={
                "mu": float(updated_a.mu),
                "sigma": float(updated_a.sigma),
                "games": rating_a.games + 1,
                "wins": rating_a.wins + (1 if decision == "A" else 0),
                "losses": rating_a.losses + (1 if decision == "B" else 0),
                "ties": rating_a.ties + (1 if decision == "TIE" else 0),
            }
        )
        next_b = rating_b.model_copy(
            update={
                "mu": float(updated_b.mu),
                "sigma": float(updated_b.sigma),
                "games": rating_b.games + 1,
                "wins": rating_b.wins + (1 if decision == "B" else 0),
                "losses": rating_b.losses + (1 if decision == "A" else 0),
                "ties": rating_b.ties + (1 if decision == "TIE" else 0),
            }
        )
        return next_a, next_b, True, decision


def apply_outcomes(
    ratings: Mapping[str, ModelRating],
    outcomes: Sequence[MatchOutcome],
    *,
    params: TrueSkillRatingParams,
    conservative_k: float = 3.0,
    elo_scale: float = 173.7178,
    invalid_policy: InvalidPolicy = "skip",
) -> RatingUpdateResult:
    """Apply a batch of outcomes to ratings in deterministic order."""
    engine = TrueSkillEngine(
        params,
        conservative_k=conservative_k,
        elo_scale=elo_scale,
    )
    next_ratings = {model_id: rating.model_copy() for model_id, rating in ratings.items()}

    processed = 0
    skipped = 0
    for outcome in outcomes:
        if outcome.model_a not in next_ratings:
            raise ValueError(f"Unknown model id in outcome.model_a: {outcome.model_a}")
        if outcome.model_b not in next_ratings:
            raise ValueError(f"Unknown model id in outcome.model_b: {outcome.model_b}")

        updated_a, updated_b, did_process, _ = engine.apply_outcome(
            next_ratings[outcome.model_a],
            next_ratings[outcome.model_b],
            decision=outcome.decision,
            invalid_policy=invalid_policy,
        )
        if did_process:
            next_ratings[outcome.model_a] = updated_a
            next_ratings[outcome.model_b] = updated_b
            processed += 1
        else:
            skipped += 1

    standings = summarize_ratings(
        next_ratings,
        params=params,
        conservative_k=conservative_k,
        elo_scale=elo_scale,
    )
    return RatingUpdateResult(
        ratings=next_ratings,
        standings=standings,
        processed_outcomes=processed,
        skipped_outcomes=skipped,
    )


def summarize_ratings(
    ratings: Mapping[str, ModelRating],
    *,
    params: TrueSkillRatingParams,
    conservative_k: float = 3.0,
    elo_scale: float = 173.7178,
) -> list[ModelStanding]:
    """Compute derived ranking fields and sort by conservative score."""
    engine = TrueSkillEngine(
        params,
        conservative_k=conservative_k,
        elo_scale=elo_scale,
    )
    standings = [
        ModelStanding(
            model_id=rating.model_id,
            mu=rating.mu,
            sigma=rating.sigma,
            games=rating.games,
            wins=rating.wins,
            losses=rating.losses,
            ties=rating.ties,
            conservative=engine.conservative_score(rating),
            elo_like=engine.elo_like(rating),
        )
        for rating in ratings.values()
    ]
    return sorted(
        standings,
        key=lambda standing: (
            -standing.conservative,
            -standing.mu,
            standing.model_id,
        ),
    )


def apply_outcomes_to_store(
    store: TournamentStore,
    config: TournamentConfig,
    outcomes: Sequence[MatchOutcome],
    *,
    step_id: int | None = None,
) -> RatingUpdateResult:
    """Apply outcomes and persist rating updates and history in the store."""
    with store.transaction():
        result = apply_outcomes(
            ratings=store.load_model_ratings(),
            outcomes=outcomes,
            params=config.rating_params,
            conservative_k=config.conservative_k,
            elo_scale=config.elo_scale,
            invalid_policy=config.invalid_policy,
        )
        for rating in result.ratings.values():
            store.upsert_model_rating(rating, commit=False)

        resolved_step_id = step_id if step_id is not None else store.next_ratings_history_step()
        store.append_ratings_history(resolved_step_id, result.ratings, commit=False)

    return result.model_copy(update={"step_id": resolved_step_id})
