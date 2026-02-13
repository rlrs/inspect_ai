from pathlib import Path

import inspect_ai.tournament.cli as tournament_cli
from inspect_ai.tournament.exports import ExportResult
from inspect_ai.tournament.orchestrator import (
    AddModelsResult,
    TournamentRunResult,
    TournamentStatus,
)


def test_cli_run_dispatches_to_run_tournament(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    config_path = tmp_path / "tournament.yaml"
    config_path.write_text("judge_model: x\n", encoding="utf-8")

    monkeypatch.setattr(
        tournament_cli,
        "run_tournament",
        lambda config, max_batches=None: _run_result(),
    )
    exit_code = tournament_cli.main(
        ["run", "--config", config_path.as_posix(), "--max-batches", "2"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Run Summary" in output
    assert "Batches Completed" in output
    assert "Tournament Status" in output


def test_cli_status_dispatches_to_tournament_status(
    monkeypatch: object, capsys: object
) -> None:
    monkeypatch.setattr(tournament_cli, "tournament_status", lambda target: _status())

    exit_code = tournament_cli.main(["status", "/tmp/state"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Tournament Status" in output
    assert "Run Status" in output
    assert "completed" in output


def test_cli_export_dispatches_to_export_rankings(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    exports_dir = tmp_path / "exports"
    monkeypatch.setattr(
        tournament_cli,
        "export_rankings",
        lambda target, output_dir=None: ExportResult(
            output_dir=exports_dir,
            rankings_json=exports_dir / "rankings.json",
            rankings_csv=exports_dir / "rankings.csv",
            pairwise_matrix_csv=exports_dir / "pairwise_matrix.csv",
        ),
    )

    exit_code = tournament_cli.main(
        ["export", "/tmp/state", "--output-dir", exports_dir.as_posix()]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Export Artifacts" in output
    assert "Rankings JSON" in output
    assert "rankings.json" in output


def test_cli_run_can_save_json_output(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    config_path = tmp_path / "tournament.yaml"
    config_path.write_text("judge_model: x\n", encoding="utf-8")
    json_out = tmp_path / "run.json"

    monkeypatch.setattr(
        tournament_cli,
        "run_tournament",
        lambda config, max_batches=None: _run_result(),
    )
    exit_code = tournament_cli.main(
        [
            "run",
            "--config",
            config_path.as_posix(),
            "--json-out",
            json_out.as_posix(),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Saved JSON output to" in output
    assert json_out.exists()


def test_cli_add_model_dispatches_to_add_models(
    monkeypatch: object, capsys: object
) -> None:
    monkeypatch.setattr(
        tournament_cli,
        "add_models",
        lambda target, models, max_batches=None: _add_models_result(models),
    )

    exit_code = tournament_cli.main(
        [
            "add-model",
            "/tmp/state",
            "--model",
            "model/c",
            "--model",
            "model/d",
            "--max-batches",
            "1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Add Models Summary" in output
    assert "Added Models" in output
    assert "model/c" in output
    assert "Tournament Status" in output


def _status() -> TournamentStatus:
    return TournamentStatus(
        project_id="project_123",
        run_status="completed",
        next_round_index=4,
        pending_batch_id=None,
        stable_batches=2,
        converged=True,
        stop_reasons=["converged"],
        total_models=2,
        total_prompts=1,
        response_count=2,
        expected_responses=2,
        missing_responses=0,
        total_matches=3,
        scheduled_matches=0,
        judged_matches=0,
        rated_matches=3,
        standings=[],
    )


def _run_result() -> TournamentRunResult:
    return TournamentRunResult(
        batches_completed=1,
        matches_scheduled=2,
        outcomes_processed=2,
        outcomes_skipped=0,
        status=_status(),
    )


def _add_models_result(models: list[str]) -> AddModelsResult:
    return AddModelsResult(
        requested_models=models,
        added_models=models,
        already_present_models=[],
        generated_models=models,
        run=_run_result(),
    )
