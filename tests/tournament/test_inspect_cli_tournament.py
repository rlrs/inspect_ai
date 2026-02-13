from click.testing import CliRunner

import inspect_ai._cli.tournament as tournament_cli
from inspect_ai._cli.main import inspect
from inspect_ai.tournament.orchestrator import TournamentRunResult, TournamentStatus


def test_inspect_tournament_run_command_dispatches(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(
        tournament_cli,
        "run_tournament",
        lambda config, max_batches=None: _run_result(),
    )

    runner = CliRunner()
    result = runner.invoke(
        inspect,
        ["tournament", "run", "tournament.yaml", "--max-batches", "1"],
    )

    assert result.exit_code == 0
    assert "Run Summary" in result.output
    assert "Batches Completed" in result.output
    assert "Tournament Status" in result.output
    assert "completed" in result.output


def test_inspect_help_lists_tournament_command() -> None:
    runner = CliRunner()
    result = runner.invoke(inspect, ["--help"])
    assert result.exit_code == 0
    assert "tournament" in result.output


def _run_result() -> TournamentRunResult:
    return TournamentRunResult(
        batches_completed=1,
        matches_scheduled=2,
        outcomes_processed=2,
        outcomes_skipped=0,
        status=TournamentStatus(
            project_id="project_123",
            run_status="completed",
            next_round_index=3,
            pending_batch_id=None,
            stable_batches=2,
            converged=True,
            stop_reasons=["converged"],
            total_models=2,
            total_prompts=1,
            response_count=2,
            expected_responses=2,
            missing_responses=0,
            total_matches=2,
            scheduled_matches=0,
            judged_matches=0,
            rated_matches=2,
            standings=[],
        ),
    )
