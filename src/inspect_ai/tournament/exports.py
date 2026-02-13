import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from .config import TournamentConfig, load_tournament_config
from .orchestrator import tournament_status
from .scorer import canonicalize_side_decision, reconcile_side_swap
from .store import TournamentStore
from .types import Decision, InvalidPolicy


class ExportResult(BaseModel):
    """Paths for generated tournament export artifacts."""

    output_dir: Path
    rankings_json: Path
    rankings_csv: Path
    pairwise_matrix_csv: Path | None = None


def export_rankings(
    config_or_state: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    include_pairwise_matrix: bool = True,
) -> ExportResult:
    """Export standings to JSON/CSV plus optional pairwise matrix CSV."""
    config = _resolve_config(config_or_state)
    status = tournament_status(config)
    state_dir = _require_state_dir(config)

    export_dir = (
        Path(output_dir) if output_dir is not None else (config.log_dir / "exports")
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    with TournamentStore(state_dir) as store:
        names_by_id = _model_names_by_id(store)
        pairwise_stats = _pairwise_stats(config, store) if include_pairwise_matrix else {}

    rankings_json = export_dir / "rankings.json"
    rankings_csv = export_dir / "rankings.csv"
    pairwise_matrix_csv = (
        export_dir / "pairwise_matrix.csv" if include_pairwise_matrix else None
    )

    ranking_rows = [
        {
            "rank": rank,
            "model_id": standing.model_id,
            "model_name": names_by_id.get(standing.model_id, standing.model_id),
            "mu": standing.mu,
            "sigma": standing.sigma,
            "conservative": standing.conservative,
            "elo_like": standing.elo_like,
            "games": standing.games,
            "wins": standing.wins,
            "losses": standing.losses,
            "ties": standing.ties,
        }
        for rank, standing in enumerate(status.standings, start=1)
    ]

    rankings_json.write_text(
        json.dumps(
            {
                "project_id": status.project_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_status": status.run_status,
                "converged": status.converged,
                "stop_reasons": status.stop_reasons,
                "total_matches": status.total_matches,
                "rated_matches": status.rated_matches,
                "models": ranking_rows,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with rankings_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "model_id",
                "model_name",
                "mu",
                "sigma",
                "conservative",
                "elo_like",
                "games",
                "wins",
                "losses",
                "ties",
            ],
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    if pairwise_matrix_csv is not None:
        _write_pairwise_matrix_csv(
            pairwise_matrix_csv,
            standings=[standing.model_id for standing in status.standings],
            names_by_id=names_by_id,
            stats=pairwise_stats,
        )

    return ExportResult(
        output_dir=export_dir,
        rankings_json=rankings_json,
        rankings_csv=rankings_csv,
        pairwise_matrix_csv=pairwise_matrix_csv,
    )


@dataclass
class _PairStats:
    wins_low: int = 0
    wins_high: int = 0
    ties: int = 0
    invalid: int = 0


def _pairwise_stats(
    config: TournamentConfig,
    store: TournamentStore,
) -> dict[tuple[str, str], _PairStats]:
    conn = store.connection()
    match_rows = conn.execute(
        """
        SELECT match_id, model_a, model_b
        FROM matches
        ORDER BY match_id
        """
    ).fetchall()
    judgment_rows = conn.execute(
        """
        SELECT match_id, side, decision
        FROM judgments
        ORDER BY match_id, side
        """
    ).fetchall()

    side_decisions: dict[str, dict[str, Decision]] = defaultdict(dict)
    for row in judgment_rows:
        match_id = str(row["match_id"])
        side = str(row["side"])
        if side not in ("ab", "ba"):
            continue
        side_decisions[match_id][side] = _as_decision(str(row["decision"]))

    pair_stats: dict[tuple[str, str], _PairStats] = {}
    for row in match_rows:
        match_id = str(row["match_id"])
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        decision = _canonical_decision_for_match(
            config.side_swap,
            config.invalid_policy,
            side_decisions.get(match_id, {}),
        )

        low, high = (model_a, model_b) if model_a <= model_b else (model_b, model_a)
        stats = pair_stats.get((low, high), _PairStats())

        if decision == "TIE":
            stats.ties += 1
        elif decision == "INVALID":
            stats.invalid += 1
        else:
            winner = model_a if decision == "A" else model_b
            if winner == low:
                stats.wins_low += 1
            else:
                stats.wins_high += 1

        pair_stats[(low, high)] = stats

    return pair_stats


def _canonical_decision_for_match(
    side_swap: bool,
    invalid_policy: InvalidPolicy,
    decisions: dict[str, Decision],
) -> Decision:
    ab = decisions.get("ab")
    ba = decisions.get("ba")
    if side_swap:
        if ab is not None and ba is not None:
            return reconcile_side_swap(ab, ba, invalid_policy=invalid_policy)
        if ab is not None:
            return canonicalize_side_decision(ab, "ab")
        if ba is not None:
            return canonicalize_side_decision(ba, "ba")
        return "INVALID"

    if ab is not None:
        return canonicalize_side_decision(ab, "ab")
    if ba is not None:
        return canonicalize_side_decision(ba, "ba")
    return "INVALID"


def _write_pairwise_matrix_csv(
    path: Path,
    *,
    standings: list[str],
    names_by_id: dict[str, str],
    stats: dict[tuple[str, str], _PairStats],
) -> None:
    columns = [names_by_id.get(model_id, model_id) for model_id in standings]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_id", "model_name", *columns])
        for row_model in standings:
            row_values: list[str] = []
            for column_model in standings:
                if row_model == column_model:
                    row_values.append("")
                    continue
                low, high = (
                    (row_model, column_model)
                    if row_model <= column_model
                    else (column_model, row_model)
                )
                pair = stats.get((low, high))
                if pair is None:
                    row_values.append("")
                    continue

                total = pair.wins_low + pair.wins_high + pair.ties
                if total == 0:
                    row_values.append("")
                    continue
                if row_model == low:
                    score = (pair.wins_low + (0.5 * pair.ties)) / total
                else:
                    score = (pair.wins_high + (0.5 * pair.ties)) / total
                row_values.append(f"{score:.6f}")

            writer.writerow(
                [row_model, names_by_id.get(row_model, row_model), *row_values]
            )


def _model_names_by_id(store: TournamentStore) -> dict[str, str]:
    rows = store.connection().execute(
        "SELECT model_id, model_name FROM models ORDER BY model_name"
    ).fetchall()
    return {str(row["model_id"]): str(row["model_name"]) for row in rows}


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


def _require_state_dir(config: TournamentConfig) -> Path:
    if config.state_dir is None:
        raise ValueError("state_dir is required")
    return config.state_dir


def _as_decision(value: str) -> Decision:
    normalized = value.strip().upper()
    if normalized in ("A", "B", "TIE", "INVALID"):
        return normalized  # type: ignore[return-value]
    return "INVALID"
