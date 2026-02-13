import json
from pathlib import Path

import click

from inspect_ai.tournament import (
    TournamentRunResult,
    export_rankings,
    resume_tournament,
    run_generation,
    run_tournament,
    tournament_status,
)
from inspect_ai.tournament.store import initialize_tournament_store


@click.group("tournament")
def tournament_command() -> None:
    """Run and manage tournament evaluations."""
    return None


@tournament_command.command("init")
@click.argument("config", type=str)
def tournament_init_command(config: str) -> None:
    """Initialize tournament state from a config file."""
    db_path = initialize_tournament_store(config)
    click.echo(db_path.as_posix())


@tournament_command.command("generate")
@click.argument("config", type=str)
@click.option(
    "--model",
    "models",
    multiple=True,
    type=str,
    help="Optional subset of contestant models to generate.",
)
def tournament_generate_command(config: str, models: tuple[str, ...]) -> None:
    """Generate contestant responses."""
    result = run_generation(config, models=list(models) if len(models) > 0 else None)
    click.echo(
        json.dumps(
            {
                "models": result.models,
                "prompt_count": result.prompt_count,
                "log_dir": result.log_dir.as_posix(),
                "log_count": result.log_count,
            },
            indent=2,
        )
    )


@tournament_command.command("run")
@click.argument("config", type=str)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional max number of batches to execute before returning.",
)
def tournament_run_command(config: str, max_batches: int | None) -> None:
    """Run a tournament from config."""
    result = run_tournament(config, max_batches=max_batches)
    click.echo(_run_result_json(result))


@tournament_command.command("resume")
@click.argument("target", type=str)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional max number of batches to execute before returning.",
)
def tournament_resume_command(target: str, max_batches: int | None) -> None:
    """Resume a tournament from config path or state directory."""
    result = resume_tournament(target, max_batches=max_batches)
    click.echo(_run_result_json(result))


@tournament_command.command("status")
@click.argument("target", type=str)
def tournament_status_command(target: str) -> None:
    """Show tournament status as JSON."""
    status = tournament_status(target)
    click.echo(status.model_dump_json(indent=2))


@tournament_command.command("export")
@click.argument("target", type=str)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Optional output directory for ranking artifacts.",
)
def tournament_export_command(target: str, output_dir: str | None) -> None:
    """Export rankings artifacts."""
    result = export_rankings(
        target,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    click.echo(
        json.dumps(
            {
                "output_dir": result.output_dir.as_posix(),
                "rankings_json": result.rankings_json.as_posix(),
                "rankings_csv": result.rankings_csv.as_posix(),
                "pairwise_matrix_csv": (
                    result.pairwise_matrix_csv.as_posix()
                    if result.pairwise_matrix_csv is not None
                    else None
                ),
            },
            indent=2,
        )
    )


def _run_result_json(result: TournamentRunResult) -> str:
    return json.dumps(
        {
            "batches_completed": result.batches_completed,
            "matches_scheduled": result.matches_scheduled,
            "outcomes_processed": result.outcomes_processed,
            "outcomes_skipped": result.outcomes_skipped,
            "status": result.status.model_dump(),
        },
        indent=2,
    )
