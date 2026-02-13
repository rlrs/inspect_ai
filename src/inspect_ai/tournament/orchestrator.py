import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from pydantic import BaseModel, Field

from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.scorer import Score

from .config import TournamentConfig, load_tournament_config
from .generation import run_generation
from .indexer import ResponseIndexReport, index_generation_responses
from .judge_task import JudgeMatch, run_judge_batch
from .rating import MatchOutcome, ModelStanding, apply_outcomes, summarize_ratings
from .scheduler import schedule_match_batch
from .scorer import canonicalize_side_decision, reconcile_side_swap
from .stopping import check_convergence, check_hard_stops_for_config
from .store import TournamentStore, initialize_tournament_store
from .types import Decision, deterministic_id

RUN_STATUS_RUNNING = "running"
RUN_STATUS_STOPPED = "stopped"
RUN_STATUS_COMPLETED = "completed"


class _ParsedJudgedSample(TypedDict):
    match_id: str
    side: str
    decision: str
    judge_model: str
    explanation: str | None
    raw_completion: str | None


class TournamentStatus(BaseModel):
    """Current state of a tournament run."""

    project_id: str | None
    run_status: str
    next_round_index: int
    pending_batch_id: str | None
    stable_batches: int
    converged: bool
    stop_reasons: list[str]
    total_models: int
    total_prompts: int
    response_count: int
    expected_responses: int
    missing_responses: int
    total_matches: int
    scheduled_matches: int
    judged_matches: int
    rated_matches: int
    min_adjacent_probability: float | None = None
    min_adjacent_margin: float | None = None
    standings: list[ModelStanding] = Field(default_factory=list)


class TournamentRunResult(BaseModel):
    """Result of running or resuming the tournament loop."""

    batches_completed: int
    matches_scheduled: int
    outcomes_processed: int
    outcomes_skipped: int
    status: TournamentStatus


def run_tournament(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    max_batches: int | None = None,
) -> TournamentRunResult:
    """Run a tournament from config."""
    parsed = load_tournament_config(config)
    initialize_tournament_store(parsed)
    state_dir = _require_state_dir(parsed)

    with TournamentStore(state_dir) as store:
        store.initialize_from_config(parsed)
        _set_run_state_defaults(store, parsed)
        _ensure_response_coverage(parsed, store, force_regenerate=parsed.regenerate_completions)
        return _run_loop(parsed, store, max_batches=max_batches)


def resume_tournament(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    max_batches: int | None = None,
) -> TournamentRunResult:
    """Resume a tournament from config or existing state directory."""
    parsed = _resolve_config(config_or_state)
    state_dir = _require_state_dir(parsed)
    with TournamentStore(state_dir) as store:
        store.initialize_from_config(parsed)
        _set_run_state_defaults(store, parsed)
        _ensure_response_coverage(parsed, store, force_regenerate=False)
        return _run_loop(parsed, store, max_batches=max_batches)


def tournament_status(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentStatus:
    """Report current tournament status."""
    parsed = _resolve_config(config_or_state)
    state_dir = _require_state_dir(parsed)
    with TournamentStore(state_dir) as store:
        return _status_from_store(parsed, store)


def _resolve_config(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
) -> TournamentConfig:
    if isinstance(config_or_state, TournamentConfig):
        return config_or_state
    if isinstance(config_or_state, Mapping):
        return TournamentConfig.model_validate(dict(config_or_state))

    try:
        return load_tournament_config(config_or_state)
    except Exception:
        state_path = Path(config_or_state)
        with TournamentStore(state_path) as store:
            config_json = store.get_run_state("config_json")
            if config_json is None or config_json.strip() == "":
                raise ValueError(
                    f"Could not load tournament config from {config_or_state!r} "
                    "and no config_json was found in run_state."
                ) from None
        return TournamentConfig.model_validate_json(config_json)


def _set_run_state_defaults(store: TournamentStore, config: TournamentConfig) -> None:
    defaults: dict[str, str] = {
        "config_json": config.model_dump_json(),
        "run_status": RUN_STATUS_RUNNING,
        "stable_batches": "0",
        "next_round_index": "1",
        "pending_batch_id": "",
        "pending_round_index": "0",
        "converged": "0",
        "stop_reasons": "[]",
        "last_min_probability": "",
        "last_min_margin": "",
    }
    with store.transaction():
        for key, value in defaults.items():
            current = store.get_run_state(key)
            if current is None:
                store.set_run_state(key, value, commit=False)


def _ensure_response_coverage(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    force_regenerate: bool,
) -> ResponseIndexReport:
    report = index_generation_responses(config, store=store)
    if force_regenerate:
        run_generation(config, models=config.contestant_models)
        report = index_generation_responses(config, store=store)
    elif report.missing_count > 0:
        missing_models = sorted(
            model_name
            for model_name, missing_prompt_ids in report.missing_by_model.items()
            if len(missing_prompt_ids) > 0
        )
        if len(missing_models) > 0:
            run_generation(config, models=missing_models)
            report = index_generation_responses(config, store=store)

    if report.missing_count > 0:
        raise RuntimeError(
            "Missing prompt/model responses after generation: "
            + json.dumps(report.missing_by_model, sort_keys=True)
        )
    return report


def _run_loop(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    max_batches: int | None,
) -> TournamentRunResult:
    batches_completed = 0
    matches_scheduled = 0
    outcomes_processed = 0
    outcomes_skipped = 0

    while True:
        pending_batch_id = _read_pending_batch_id(store)
        pending_round_index = _read_int_state(
            store, "pending_round_index", default=_read_int_state(store, "next_round_index", default=1)
        )

        if pending_batch_id is None:
            total_matches = store.table_count("matches")
            if total_matches >= config.max_total_matches:
                _record_stop(
                    store,
                    run_status=RUN_STATUS_COMPLETED,
                    reasons=["max_total_matches_reached"],
                )
                break

            round_index = _read_int_state(store, "next_round_index", default=1)
            batch_id = f"batch-{round_index:06d}"
            schedule_result = schedule_match_batch(
                config,
                store,
                batch_id=batch_id,
                round_index=round_index,
                seed=config.seed,
                persist=True,
            )
            if schedule_result.exhausted:
                _record_stop(
                    store,
                    run_status=RUN_STATUS_COMPLETED,
                    reasons=["no_eligible_pairs"],
                )
                break

            with store.transaction():
                store.set_run_state("pending_batch_id", batch_id, commit=False)
                store.set_run_state("pending_round_index", str(round_index), commit=False)
                store.set_run_state("run_status", RUN_STATUS_RUNNING, commit=False)
            pending_batch_id = batch_id
            pending_round_index = round_index
            matches_scheduled += len(schedule_result.scheduled)

        judged_now = _judge_pending_matches(config, store, pending_batch_id)
        del judged_now

        applied = _apply_pending_batch_outcomes(
            config,
            store,
            batch_id=pending_batch_id,
            round_index=pending_round_index,
        )
        if applied is None:
            _clear_pending_batch(store)
            continue

        batches_completed += 1
        outcomes_processed += applied["processed"]
        outcomes_skipped += applied["skipped"]

        if max_batches is not None and batches_completed >= max_batches:
            break

        if applied["converged"]:
            _record_stop(
                store,
                run_status=RUN_STATUS_COMPLETED,
                reasons=["converged"],
            )
            break

        total_matches_after = store.table_count("matches")
        hard_stop = check_hard_stops_for_config(
            config,
            total_matches=total_matches_after,
            no_eligible_pairs=False,
        )
        if hard_stop.should_stop:
            _record_stop(
                store,
                run_status=RUN_STATUS_COMPLETED,
                reasons=hard_stop.reasons,
            )
            break

    status = _status_from_store(config, store)
    return TournamentRunResult(
        batches_completed=batches_completed,
        matches_scheduled=matches_scheduled,
        outcomes_processed=outcomes_processed,
        outcomes_skipped=outcomes_skipped,
        status=status,
    )


def _judge_pending_matches(
    config: TournamentConfig,
    store: TournamentStore,
    batch_id: str,
) -> int:
    rows = store.load_batch_matches(batch_id, statuses=["scheduled"])
    if len(rows) == 0:
        return 0

    matches = [
        JudgeMatch(
            match_id=str(row["match_id"]),
            prompt_id=str(row["prompt_id"]),
            prompt=str(row["prompt_text"]),
            model_a=str(row["model_a_name"]),
            model_b=str(row["model_b_name"]),
            model_a_id=str(row["model_a_id"]),
            model_b_id=str(row["model_b_id"]),
            response_a=str(row["response_a_text"]),
            response_b=str(row["response_b_text"]),
        )
        for row in rows
    ]

    side_count = 2 if config.side_swap else 1
    matches_per_judge_run = max(1, config.judge_max_samples // side_count)
    for chunk in _chunked(matches, size=matches_per_judge_run):
        judge_result = run_judge_batch(
            config,
            chunk,
            log_dir=config.log_dir / "judge",
        )
        _ingest_judge_logs(config, store, judge_result.logs)

    with store.transaction():
        store.set_batch_match_status(
            batch_id=batch_id,
            status="judged",
            from_statuses=["scheduled"],
            commit=False,
        )
    return len(matches)


def _ingest_judge_logs(
    config: TournamentConfig,
    store: TournamentStore,
    logs: Sequence[EvalLog],
) -> None:
    with store.transaction():
        for log in logs:
            source_log = _relative_log_name(config.log_dir / "judge", log.location)
            for sample in (log.samples or []):
                parsed = _parse_judged_sample(sample)
                if parsed is None:
                    continue
                judgment_id = deterministic_id(
                    "judgment",
                    parsed["match_id"],
                    parsed["side"],
                    length=20,
                )
                store.upsert_judgment(
                    judgment_id=judgment_id,
                    match_id=parsed["match_id"],
                    side=parsed["side"],
                    decision=parsed["decision"],
                    judge_model=parsed["judge_model"],
                    explanation=parsed["explanation"],
                    raw_completion=parsed["raw_completion"],
                    source_log=source_log,
                    sample_uuid=sample.uuid,
                    commit=False,
                )


def _parse_judged_sample(sample: EvalSample) -> _ParsedJudgedSample | None:
    metadata = sample.metadata if sample.metadata is not None else {}
    score = _extract_judge_score(sample.scores)
    if score is None:
        return None

    match_id = metadata.get("match_id")
    if not isinstance(match_id, str) or match_id.strip() == "":
        return None

    side = metadata.get("side")
    if not isinstance(side, str) or side not in ("ab", "ba"):
        side = "ab"

    score_metadata = score.metadata if isinstance(score.metadata, dict) else {}
    judge_model = score_metadata.get("judge_model")
    decision = _as_decision(score.value)
    return {
        "match_id": match_id,
        "side": side,
        "decision": decision,
        "judge_model": str(judge_model) if judge_model is not None else "",
        "explanation": score.explanation,
        "raw_completion": score.explanation,
    }


def _extract_judge_score(scores: dict[str, Score] | None) -> Score | None:
    if scores is None or len(scores) == 0:
        return None
    if "pairwise_judge" in scores:
        return scores["pairwise_judge"]
    return next(iter(scores.values()))


def _apply_pending_batch_outcomes(
    config: TournamentConfig,
    store: TournamentStore,
    *,
    batch_id: str,
    round_index: int,
) -> dict[str, int | bool] | None:
    rows_to_rate = store.load_batch_matches(batch_id, statuses=["judged", "scheduled"])
    if len(rows_to_rate) == 0:
        return None

    outcomes = _canonical_outcomes_for_batch(config, store, batch_id)
    stable_batches = _read_int_state(store, "stable_batches", default=0)

    with store.transaction():
        rating_result = apply_outcomes(
            ratings=store.load_model_ratings(),
            outcomes=outcomes,
            params=config.rating_params,
            conservative_k=config.conservative_k,
            elo_scale=config.elo_scale,
            invalid_policy=config.invalid_policy,
        )
        for rating in rating_result.ratings.values():
            store.upsert_model_rating(rating, commit=False)

        step_id = store.next_ratings_history_step()
        store.append_ratings_history(step_id, rating_result.ratings, commit=False)
        store.set_batch_match_status(
            batch_id=batch_id,
            status="rated",
            from_statuses=["scheduled", "judged"],
            commit=False,
        )

        convergence = check_convergence(
            rating_result.ratings,
            rating_params=config.rating_params,
            p_stop=config.p_stop,
            epsilon=config.epsilon,
            conservative_k=config.conservative_k,
            n_stable_batches=config.n_stable_batches,
            stable_batches=stable_batches,
            elo_scale=config.elo_scale,
        )
        store.set_run_state("stable_batches", str(convergence.stable_batches), commit=False)
        store.set_run_state("converged", "1" if convergence.converged else "0", commit=False)
        store.set_run_state("pending_batch_id", "", commit=False)
        store.set_run_state("pending_round_index", "0", commit=False)
        store.set_run_state("next_round_index", str(round_index + 1), commit=False)
        store.set_run_state("last_batch_id", batch_id, commit=False)
        store.set_run_state(
            "last_min_probability",
            "" if convergence.min_probability is None else str(convergence.min_probability),
            commit=False,
        )
        store.set_run_state(
            "last_min_margin",
            "" if convergence.min_margin is None else str(convergence.min_margin),
            commit=False,
        )
        store.set_run_state("run_status", RUN_STATUS_RUNNING, commit=False)

    return {
        "processed": rating_result.processed_outcomes,
        "skipped": rating_result.skipped_outcomes,
        "converged": convergence.converged,
    }


def _canonical_outcomes_for_batch(
    config: TournamentConfig,
    store: TournamentStore,
    batch_id: str,
) -> list[MatchOutcome]:
    rows = store.load_batch_judgments(batch_id)
    decisions_by_match: dict[str, dict[str, Decision]] = defaultdict(dict)
    model_pair_by_match: dict[str, tuple[str, str]] = {}

    for row in rows:
        match = str(row["match_id"])
        model_pair_by_match[match] = (str(row["model_a"]), str(row["model_b"]))

        side = row["side"]
        decision = row["decision"]
        if side is None or decision is None:
            continue
        side_value = str(side)
        if side_value not in ("ab", "ba"):
            continue
        decisions_by_match[match][side_value] = _as_decision(str(decision))

    outcomes: list[MatchOutcome] = []
    for match in sorted(model_pair_by_match):
        model_a, model_b = model_pair_by_match[match]
        side_decisions = decisions_by_match.get(match, {})
        ab = side_decisions.get("ab")
        ba = side_decisions.get("ba")
        if config.side_swap:
            if ab is not None and ba is not None:
                decision = reconcile_side_swap(ab, ba, invalid_policy=config.invalid_policy)
            elif ab is not None:
                decision = canonicalize_side_decision(ab, "ab")
            elif ba is not None:
                decision = canonicalize_side_decision(ba, "ba")
            else:
                decision = "INVALID"
        else:
            if ab is not None:
                decision = canonicalize_side_decision(ab, "ab")
            elif ba is not None:
                decision = canonicalize_side_decision(ba, "ba")
            else:
                decision = "INVALID"

        outcomes.append(
            MatchOutcome(
                model_a=model_a,
                model_b=model_b,
                decision=decision,
            )
        )
    return outcomes


def _status_from_store(config: TournamentConfig, store: TournamentStore) -> TournamentStatus:
    ratings = store.load_model_ratings()
    names_by_id = store.model_names_by_id()
    standings = summarize_ratings(
        ratings,
        params=config.rating_params,
        conservative_k=config.conservative_k,
        elo_scale=config.elo_scale,
    )
    standings = [
        standing.model_copy(
            update={"model_name": names_by_id.get(standing.model_id, standing.model_id)}
        )
        for standing in standings
    ]
    expected_responses = len(config.contestant_models) * len(config.prompts)
    response_count = store.table_count("responses")
    missing = max(0, expected_responses - response_count)
    status_counts = store.match_status_counts()

    run_status = store.get_run_state("run_status") or RUN_STATUS_RUNNING
    stop_reasons_raw = store.get_run_state("stop_reasons") or "[]"
    try:
        stop_reasons = json.loads(stop_reasons_raw)
        if not isinstance(stop_reasons, list):
            stop_reasons = []
    except json.JSONDecodeError:
        stop_reasons = []

    return TournamentStatus(
        project_id=store.get_run_state("project_id"),
        run_status=run_status,
        next_round_index=_read_int_state(store, "next_round_index", default=1),
        pending_batch_id=_read_pending_batch_id(store),
        stable_batches=_read_int_state(store, "stable_batches", default=0),
        converged=_read_bool_state(store, "converged", default=False),
        stop_reasons=[str(reason) for reason in stop_reasons],
        total_models=len(config.contestant_models),
        total_prompts=len(config.prompts),
        response_count=response_count,
        expected_responses=expected_responses,
        missing_responses=missing,
        total_matches=store.table_count("matches"),
        scheduled_matches=status_counts.get("scheduled", 0),
        judged_matches=status_counts.get("judged", 0),
        rated_matches=status_counts.get("rated", 0),
        min_adjacent_probability=_read_float_state(store, "last_min_probability"),
        min_adjacent_margin=_read_float_state(store, "last_min_margin"),
        standings=standings,
    )


def _record_stop(store: TournamentStore, *, run_status: str, reasons: list[str]) -> None:
    with store.transaction():
        store.set_run_state("run_status", run_status, commit=False)
        store.set_run_state("stop_reasons", json.dumps(reasons, sort_keys=True), commit=False)
        if "converged" in reasons:
            store.set_run_state("converged", "1", commit=False)


def _clear_pending_batch(store: TournamentStore) -> None:
    with store.transaction():
        store.set_run_state("pending_batch_id", "", commit=False)
        store.set_run_state("pending_round_index", "0", commit=False)


def _read_pending_batch_id(store: TournamentStore) -> str | None:
    batch_id = store.get_run_state("pending_batch_id")
    if batch_id is None:
        return None
    value = batch_id.strip()
    return value if value != "" else None


def _read_int_state(store: TournamentStore, key: str, *, default: int) -> int:
    value = store.get_run_state(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_bool_state(store: TournamentStore, key: str, *, default: bool) -> bool:
    value = store.get_run_state(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _read_float_state(store: TournamentStore, key: str) -> float | None:
    value = store.get_run_state(key)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_decision(value: Any) -> Decision:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in ("A", "B", "TIE", "INVALID"):
            return normalized  # type: ignore[return-value]
    return "INVALID"


def _relative_log_name(base_dir: Path, log_name: str | None) -> str | None:
    if log_name is None or "://" in log_name:
        return log_name
    try:
        return Path(log_name).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return log_name


def _chunked(items: Sequence[JudgeMatch], *, size: int) -> Sequence[Sequence[JudgeMatch]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def _require_state_dir(config: TournamentConfig) -> Path:
    if config.state_dir is None:
        raise ValueError("state_dir is required")
    return config.state_dir
