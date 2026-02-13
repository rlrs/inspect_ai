import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

from pydantic import BaseModel

from .config import TournamentConfig, load_tournament_config
from .rating import TrueSkillEngine, summarize_ratings
from .store import TournamentStore
from .types import match_id


class ScheduledMatch(BaseModel):
    """One scheduled pairwise match."""

    match_id: str
    model_a: str
    model_b: str
    prompt_id: str
    response_a_id: str
    response_b_id: str
    batch_id: str
    round_index: int
    priority: float
    forced: bool
    pair_matches: int
    prompt_uses: int


class ScheduleBatchResult(BaseModel):
    """Scheduler output for one batch."""

    batch_id: str
    round_index: int
    scheduled: list[ScheduledMatch]
    candidate_pairs: int

    @property
    def exhausted(self) -> bool:
        """Whether no matches were schedulable."""
        return len(self.scheduled) == 0


@dataclass(frozen=True)
class ScheduleWeights:
    entropy: float = 1.0
    uncertainty: float = 1.0
    focus: float = 0.5
    saturation: float = 0.5
    force_bonus: float = 1000.0


@dataclass(frozen=True)
class _Candidate:
    priority: float
    tie_break: str
    model_a: str
    model_b: str
    prompt_id: str
    response_a_id: str
    response_b_id: str
    forced: bool
    pair_matches: int
    prompt_uses: int


def schedule_match_batch(
    config: TournamentConfig | Mapping[str, Any] | str,
    store: TournamentStore,
    *,
    batch_id: str,
    round_index: int,
    seed: int | None = None,
    persist: bool = False,
    weights: ScheduleWeights = ScheduleWeights(),
) -> ScheduleBatchResult:
    """Schedule one active-learning batch of matches."""
    parsed = load_tournament_config(config)
    rng = random.Random(seed if seed is not None else parsed.seed + round_index)
    conn = store.connection()

    ratings = store.load_model_ratings()
    if len(ratings) < 2:
        return ScheduleBatchResult(
            batch_id=batch_id,
            round_index=round_index,
            scheduled=[],
            candidate_pairs=0,
        )

    responses = _load_responses(conn)
    pair_counts, prompt_counts = _load_match_counts(conn)
    prompt_ids = [prompt.id for prompt in parsed.prompts]

    standings = summarize_ratings(
        ratings,
        params=parsed.rating_params,
        conservative_k=parsed.conservative_k,
        elo_scale=parsed.elo_scale,
    )
    rank_by_model = {standing.model_id: index for index, standing in enumerate(standings)}
    engine = TrueSkillEngine(
        parsed.rating_params,
        conservative_k=parsed.conservative_k,
        elo_scale=parsed.elo_scale,
    )

    candidates: list[_Candidate] = []
    model_ids = sorted(ratings.keys())
    for model_a, model_b in combinations(model_ids, 2):
        pair_key = _canonical_pair(model_a, model_b)
        pair_match_count = pair_counts.get(pair_key, 0)
        if pair_match_count >= parsed.max_pair_matches:
            continue

        prompt_choice = _choose_prompt_for_pair(
            model_a=model_a,
            model_b=model_b,
            prompt_ids=prompt_ids,
            responses=responses,
            prompt_counts=prompt_counts,
            max_prompt_uses_per_pair=parsed.max_prompt_uses_per_pair,
            rng=rng,
        )
        if prompt_choice is None:
            continue

        prompt_id, prompt_uses = prompt_choice
        response_a_id = responses[(model_a, prompt_id)]
        response_b_id = responses[(model_b, prompt_id)]

        rating_a = ratings[model_a]
        rating_b = ratings[model_b]
        p_ab = engine.win_probability(rating_a, rating_b)
        entropy = _entropy(p_ab)
        uncertainty = rating_a.sigma + rating_b.sigma
        focus = _focus_score(
            rank_by_model.get(model_a, 0),
            rank_by_model.get(model_b, 0),
            top_k=parsed.top_k_focus,
        )
        saturation = pair_match_count / float(max(1, parsed.max_pair_matches))
        forced = pair_match_count < parsed.min_pair_matches

        priority = (
            (weights.entropy * entropy)
            + (weights.uncertainty * uncertainty)
            + (weights.focus * focus)
            - (weights.saturation * saturation)
        )
        if forced:
            priority += weights.force_bonus

        candidates.append(
            _Candidate(
                priority=priority,
                tie_break=f"{model_a}:{model_b}:{prompt_id}",
                model_a=model_a,
                model_b=model_b,
                prompt_id=prompt_id,
                response_a_id=response_a_id,
                response_b_id=response_b_id,
                forced=forced,
                pair_matches=pair_match_count,
                prompt_uses=prompt_uses,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.priority,
            candidate.tie_break,
        )
    )
    selected = candidates[: parsed.batch_size]

    scheduled = [
        ScheduledMatch(
            match_id=match_id(
                candidate.model_a,
                candidate.model_b,
                candidate.prompt_id,
                round_index,
                batch_id,
            ),
            model_a=candidate.model_a,
            model_b=candidate.model_b,
            prompt_id=candidate.prompt_id,
            response_a_id=candidate.response_a_id,
            response_b_id=candidate.response_b_id,
            batch_id=batch_id,
            round_index=round_index,
            priority=candidate.priority,
            forced=candidate.forced,
            pair_matches=candidate.pair_matches,
            prompt_uses=candidate.prompt_uses,
        )
        for candidate in selected
    ]

    if persist and len(scheduled) > 0:
        with store.transaction():
            for scheduled_match in scheduled:
                store.upsert_match(
                    match_id=scheduled_match.match_id,
                    model_a=scheduled_match.model_a,
                    model_b=scheduled_match.model_b,
                    prompt_id=scheduled_match.prompt_id,
                    response_a_id=scheduled_match.response_a_id,
                    response_b_id=scheduled_match.response_b_id,
                    batch_id=scheduled_match.batch_id,
                    round_index=scheduled_match.round_index,
                    status="scheduled",
                    commit=False,
                )

    return ScheduleBatchResult(
        batch_id=batch_id,
        round_index=round_index,
        scheduled=scheduled,
        candidate_pairs=len(candidates),
    )


def _load_responses(conn: Any) -> dict[tuple[str, str], str]:
    rows = conn.execute(
        "SELECT model_id, prompt_id, response_id FROM responses"
    ).fetchall()
    return {
        (str(row["model_id"]), str(row["prompt_id"])): str(row["response_id"])
        for row in rows
    }


def _load_match_counts(
    conn: Any,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str, str], int]]:
    pair_counts: dict[tuple[str, str], int] = {}
    prompt_counts: dict[tuple[str, str, str], int] = {}

    rows = conn.execute(
        "SELECT model_a, model_b, prompt_id FROM matches"
    ).fetchall()
    for row in rows:
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        prompt_id = str(row["prompt_id"])
        pair_key = _canonical_pair(model_a, model_b)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        prompt_key = (pair_key[0], pair_key[1], prompt_id)
        prompt_counts[prompt_key] = prompt_counts.get(prompt_key, 0) + 1

    return pair_counts, prompt_counts


def _choose_prompt_for_pair(
    *,
    model_a: str,
    model_b: str,
    prompt_ids: list[str],
    responses: dict[tuple[str, str], str],
    prompt_counts: dict[tuple[str, str, str], int],
    max_prompt_uses_per_pair: int,
    rng: random.Random,
) -> tuple[str, int] | None:
    pair_key = _canonical_pair(model_a, model_b)
    candidates: list[tuple[int, str]] = []

    for prompt_id in prompt_ids:
        if (model_a, prompt_id) not in responses or (model_b, prompt_id) not in responses:
            continue
        uses = prompt_counts.get((pair_key[0], pair_key[1], prompt_id), 0)
        if uses < max_prompt_uses_per_pair:
            candidates.append((uses, prompt_id))

    if len(candidates) == 0:
        return None

    min_uses = min(uses for uses, _ in candidates)
    min_prompts = sorted(prompt_id for uses, prompt_id in candidates if uses == min_uses)
    chosen_prompt = rng.choice(min_prompts)
    return chosen_prompt, min_uses


def _canonical_pair(model_a: str, model_b: str) -> tuple[str, str]:
    return (model_a, model_b) if model_a <= model_b else (model_b, model_a)


def _entropy(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p)) - ((1.0 - p) * math.log(1.0 - p))


def _focus_score(rank_a: int, rank_b: int, *, top_k: int) -> float:
    if top_k <= 0:
        return 0.0

    in_top_a = rank_a < top_k
    in_top_b = rank_b < top_k
    if in_top_a and in_top_b:
        return 1.0
    if in_top_a or in_top_b:
        return 0.75
    if abs(rank_a - rank_b) <= 1:
        return 0.25
    return 0.0
