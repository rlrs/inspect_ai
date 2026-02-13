from pathlib import Path

import click

from inspect_ai.tournament import (
    add_models,
    export_rankings,
    resume_tournament,
    run_generation,
    run_tournament,
    tournament_status,
)
from inspect_ai.tournament._cli_format import (
    add_models_result_payload,
    export_result_payload,
    format_add_models_result,
    format_export_result,
    format_generation_result,
    format_run_result,
    format_status,
    generation_result_payload,
    run_result_payload,
    status_payload,
    write_json_output,
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
@click.option(
    "--json-out",
    type=str,
    default=None,
    help="Optional path to save JSON output.",
)
def tournament_generate_command(
    config: str, models: tuple[str, ...], json_out: str | None
) -> None:
    """Generate contestant responses."""
    result = run_generation(config, models=list(models) if len(models) > 0 else None)
    payload = generation_result_payload(result)
    output_path = write_json_output(payload, json_out)
    click.echo(format_generation_result(result))
    if output_path is not None:
        click.echo(f"\nSaved JSON output to {output_path.as_posix()}")


@tournament_command.command("run")
@click.argument("config", type=str)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional max number of batches to execute before returning.",
)
@click.option(
    "--json-out",
    type=str,
    default=None,
    help="Optional path to save JSON output.",
)
def tournament_run_command(
    config: str, max_batches: int | None, json_out: str | None
) -> None:
    """Run a tournament from config."""
    result = run_tournament(config, max_batches=max_batches)
    payload = run_result_payload(result)
    output_path = write_json_output(payload, json_out)
    click.echo(format_run_result(result))
    if output_path is not None:
        click.echo(f"\nSaved JSON output to {output_path.as_posix()}")


@tournament_command.command("resume")
@click.argument("target", type=str)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional max number of batches to execute before returning.",
)
@click.option(
    "--json-out",
    type=str,
    default=None,
    help="Optional path to save JSON output.",
)
def tournament_resume_command(
    target: str, max_batches: int | None, json_out: str | None
) -> None:
    """Resume a tournament from config path or state directory."""
    result = resume_tournament(target, max_batches=max_batches)
    payload = run_result_payload(result)
    output_path = write_json_output(payload, json_out)
    click.echo(format_run_result(result))
    if output_path is not None:
        click.echo(f"\nSaved JSON output to {output_path.as_posix()}")


@tournament_command.command("add-model")
@click.argument("target", type=str)
@click.option(
    "--model",
    "models",
    multiple=True,
    required=True,
    type=str,
    help="Model name to add (repeat for multiple models).",
)
@click.option(
    "--max-batches",
    type=int,
    default=None,
    help="Optional max number of batches to execute before returning.",
)
@click.option(
    "--json-out",
    type=str,
    default=None,
    help="Optional path to save JSON output.",
)
def tournament_add_model_command(
    target: str,
    models: tuple[str, ...],
    max_batches: int | None,
    json_out: str | None,
) -> None:
    """Add one or more models to an existing tournament."""
    result = add_models(target, models=list(models), max_batches=max_batches)
    payload = add_models_result_payload(result)
    output_path = write_json_output(payload, json_out)
    click.echo(format_add_models_result(result))
    if output_path is not None:
        click.echo(f"\nSaved JSON output to {output_path.as_posix()}")


@tournament_command.command("status")
@click.argument("target", type=str)
@click.option(
    "--json-out",
    type=str,
    default=None,
    help="Optional path to save JSON output.",
)
def tournament_status_command(target: str, json_out: str | None) -> None:
    """Show tournament status."""
    status = tournament_status(target)
    payload = status_payload(status)
    output_path = write_json_output(payload, json_out)
    click.echo(format_status(status))
    if output_path is not None:
        click.echo(f"\nSaved JSON output to {output_path.as_posix()}")


@tournament_command.command("export")
@click.argument("target", type=str)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Optional output directory for ranking artifacts.",
)
@click.option(
    "--json-out",
    type=str,
    default=None,
    help="Optional path to save JSON output.",
)
def tournament_export_command(
    target: str, output_dir: str | None, json_out: str | None
) -> None:
    """Export rankings artifacts."""
    result = export_rankings(
        target,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    payload = export_result_payload(result)
    output_path = write_json_output(payload, json_out)
    click.echo(format_export_result(result))
    if output_path is not None:
        click.echo(f"\nSaved JSON output to {output_path.as_posix()}")
