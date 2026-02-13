from pathlib import Path

from inspect_ai.tournament import (
    TournamentConfig,
    TournamentStore,
    index_generation_responses,
    initialize_tournament_store,
    run_generation,
)


def test_generation_and_indexing_end_to_end(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    initialize_tournament_store(config)

    result = run_generation(config)
    assert set(result.models) == {"mockllm/model", "mockllm/model2"}
    assert result.prompt_count == 2
    assert result.log_count >= 2

    report = index_generation_responses(config)
    assert report.log_errors == 0
    assert report.responses_inserted == 4
    assert report.missing_count == 0

    with TournamentStore(config.state_dir) as store:
        assert store.table_count("responses") == 4


def test_indexing_reports_missing_coverage_and_is_idempotent(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    initialize_tournament_store(config)

    run_generation(config, models=["mockllm/model"])
    first_report = index_generation_responses(config)

    assert first_report.responses_inserted == 2
    assert first_report.missing_by_model["mockllm/model"] == []
    assert first_report.missing_by_model["mockllm/model2"] == ["p-1", "p-2"]

    second_report = index_generation_responses(config)
    assert second_report.responses_inserted == 0
    assert second_report.missing_by_model == first_report.missing_by_model

    with TournamentStore(config.state_dir) as store:
        assert store.table_count("responses") == 2


def test_generation_runs_eval_set_per_model(tmp_path: Path, monkeypatch) -> None:
    config = build_config(tmp_path)
    calls: list[str] = []

    def fake_eval_set(**kwargs):
        calls.append(str(kwargs["model"]))
        return True, []

    monkeypatch.setattr("inspect_ai.tournament.generation.eval_set", fake_eval_set)

    result = run_generation(config)

    assert result.models == ["mockllm/model", "mockllm/model2"]
    assert calls == ["mockllm/model", "mockllm/model2"]


def build_config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": tmp_path / "logs" / "completions",
            "log_dir": tmp_path / "logs" / "tournament",
            "state_dir": tmp_path / "logs" / "tournament" / "state",
            "contestant_models": ["mockllm/model", "mockllm/model2"],
            "prompts": [
                {"id": "p-1", "text": "Write one short sentence about the moon."},
                {"id": "p-2", "text": "Write one short sentence about the sea."},
            ],
            "contestant_generate_config": {"max_tokens": 64},
            "judge_model": "mockllm/model",
            "judge_prompt_template": "Prompt: {prompt}",
        }
    )
