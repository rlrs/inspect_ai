import json
from pathlib import Path
from typing import Any, Sequence

from .exports import ExportResult
from .generation import GenerationRunResult
from .orchestrator import AddModelsResult, TournamentRunResult, TournamentStatus
from .rating import ModelStanding


def generation_result_payload(result: GenerationRunResult) -> dict[str, Any]:
    return {
        "models": result.models,
        "prompt_count": result.prompt_count,
        "log_dir": result.log_dir.as_posix(),
        "log_count": result.log_count,
    }


def run_result_payload(result: TournamentRunResult) -> dict[str, Any]:
    return {
        "batches_completed": result.batches_completed,
        "matches_scheduled": result.matches_scheduled,
        "outcomes_processed": result.outcomes_processed,
        "outcomes_skipped": result.outcomes_skipped,
        "status": status_payload(result.status),
    }


def add_models_result_payload(result: AddModelsResult) -> dict[str, Any]:
    return {
        "requested_models": result.requested_models,
        "added_models": result.added_models,
        "already_present_models": result.already_present_models,
        "generated_models": result.generated_models,
        "run": run_result_payload(result.run),
    }


def status_payload(status: TournamentStatus) -> dict[str, Any]:
    return status.model_dump()


def export_result_payload(result: ExportResult) -> dict[str, Any]:
    return {
        "output_dir": result.output_dir.as_posix(),
        "rankings_json": result.rankings_json.as_posix(),
        "rankings_csv": result.rankings_csv.as_posix(),
        "pairwise_matrix_csv": (
            result.pairwise_matrix_csv.as_posix()
            if result.pairwise_matrix_csv is not None
            else None
        ),
    }


def write_json_output(payload: dict[str, Any], json_out: str | Path | None) -> Path | None:
    if json_out is None:
        return None

    output_path = Path(json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def format_generation_result(result: GenerationRunResult) -> str:
    lines: list[str] = []
    lines.append(
        _format_key_value_table(
            "Generation Summary",
            [
                ("Prompt Count", str(result.prompt_count)),
                ("Model Count", str(len(result.models))),
                ("Log Count", str(result.log_count)),
                ("Log Dir", result.log_dir.as_posix()),
            ],
        )
    )

    model_rows = [[str(index), model] for index, model in enumerate(result.models, 1)]
    lines.append(
        _format_table(
            "Generated Models",
            ["#", "Model Name"],
            model_rows,
            align_right={0},
        )
    )
    return "\n\n".join(lines)


def format_run_result(result: TournamentRunResult) -> str:
    lines: list[str] = []
    lines.append(
        _format_key_value_table(
            "Run Summary",
            [
                ("Batches Completed", str(result.batches_completed)),
                ("Matches Scheduled", str(result.matches_scheduled)),
                ("Outcomes Processed", str(result.outcomes_processed)),
                ("Outcomes Skipped", str(result.outcomes_skipped)),
            ],
        )
    )
    lines.append(format_status(result.status, title="Tournament Status"))
    return "\n\n".join(lines)


def format_add_models_result(result: AddModelsResult) -> str:
    lines: list[str] = []
    lines.append(
        _format_key_value_table(
            "Add Models Summary",
            [
                ("Requested", str(len(result.requested_models))),
                ("Added", str(len(result.added_models))),
                ("Already Present", str(len(result.already_present_models))),
                ("Generated", str(len(result.generated_models))),
            ],
        )
    )
    lines.append(
        _format_table(
            "Added Models",
            ["#", "Model Name"],
            [[str(index), name] for index, name in enumerate(result.added_models, start=1)],
            align_right={0},
        )
        if len(result.added_models) > 0
        else "Added Models\n(no rows)"
    )
    lines.append(format_run_result(result.run))
    return "\n\n".join(lines)


def format_status(status: TournamentStatus, *, title: str = "Tournament Status") -> str:
    stop_reasons = ", ".join(status.stop_reasons) if status.stop_reasons else "-"
    lines: list[str] = []
    lines.append(
        _format_key_value_table(
            title,
            [
                ("Project ID", status.project_id or "-"),
                ("Run Status", status.run_status),
                ("Converged", str(status.converged)),
                ("Stable Batches", str(status.stable_batches)),
                ("Stop Reasons", stop_reasons),
                ("Next Round Index", str(status.next_round_index)),
                ("Pending Batch ID", status.pending_batch_id or "-"),
                ("Total Models", str(status.total_models)),
                ("Total Prompts", str(status.total_prompts)),
                ("Responses", f"{status.response_count}/{status.expected_responses}"),
                ("Missing Responses", str(status.missing_responses)),
                ("Total Matches", str(status.total_matches)),
                ("Rated Matches", str(status.rated_matches)),
                ("Judged Matches", str(status.judged_matches)),
                ("Scheduled Matches", str(status.scheduled_matches)),
            ],
        )
    )

    standings_rows = _standings_rows(status.standings)
    if len(standings_rows) > 0:
        lines.append(
            _format_table(
                "Standings",
                [
                    "Rank",
                    "Model Name",
                    "Mu",
                    "Sigma",
                    "Conservative",
                    "Elo-like",
                    "Games",
                    "W",
                    "L",
                    "T",
                ],
                standings_rows,
                align_right={0, 2, 3, 4, 5, 6, 7, 8, 9},
            )
        )
    else:
        lines.append("Standings\n(no rows)")

    return "\n\n".join(lines)


def format_export_result(result: ExportResult) -> str:
    return _format_key_value_table(
        "Export Artifacts",
        [
            ("Output Dir", result.output_dir.as_posix()),
            ("Rankings JSON", result.rankings_json.as_posix()),
            ("Rankings CSV", result.rankings_csv.as_posix()),
            (
                "Pairwise Matrix CSV",
                result.pairwise_matrix_csv.as_posix()
                if result.pairwise_matrix_csv is not None
                else "-",
            ),
        ],
    )


def _standings_rows(standings: Sequence[ModelStanding]) -> list[list[str]]:
    rows: list[list[str]] = []
    for rank, standing in enumerate(standings, start=1):
        rows.append(
            [
                str(rank),
                standing.model_name or standing.model_id,
                f"{standing.mu:.3f}",
                f"{standing.sigma:.3f}",
                f"{standing.conservative:.3f}",
                f"{standing.elo_like:.1f}",
                str(standing.games),
                str(standing.wins),
                str(standing.losses),
                str(standing.ties),
            ]
        )
    return rows


def _format_key_value_table(title: str, rows: Sequence[tuple[str, str]]) -> str:
    table_rows = [[key, value] for key, value in rows]
    return _format_table(title, ["Field", "Value"], table_rows)


def _format_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    align_right: set[int] | None = None,
) -> str:
    normalized_columns = [str(column) for column in columns]
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    if any(len(row) != len(normalized_columns) for row in normalized_rows):
        raise ValueError("all rows must have the same number of columns as headers")

    align_right = align_right if align_right is not None else set()
    widths = [len(column) for column in normalized_columns]
    for row in normalized_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_cell(index: int, value: str) -> str:
        if index in align_right:
            return value.rjust(widths[index])
        return value.ljust(widths[index])

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = "| " + " | ".join(
        normalized_columns[index].ljust(widths[index])
        for index in range(len(normalized_columns))
    ) + " |"
    body = [
        "| "
        + " | ".join(format_cell(index, value) for index, value in enumerate(row))
        + " |"
        for row in normalized_rows
    ]

    return "\n".join([title, separator, header, separator, *body, separator])
