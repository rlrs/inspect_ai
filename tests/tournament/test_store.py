from pathlib import Path

from inspect_ai.tournament import TournamentConfig
from inspect_ai.tournament.store import TournamentStore, initialize_tournament_store
from inspect_ai.tournament.types import model_id


def test_initialize_tournament_store_creates_schema_and_seed_data(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path)
    db_path = initialize_tournament_store(config)

    assert db_path.exists()
    with TournamentStore(db_path) as store:
        assert store.schema_version == 1
        assert store.table_count("models") == 2
        assert store.table_count("prompts") == 2
        assert store.table_count("ratings") == 2
        assert store.get_run_state("seed") == str(config.seed)
        assert store.get_run_state("project_id") is not None


def test_store_initialization_is_idempotent(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    with TournamentStore(config.state_dir) as store:
        store.initialize_from_config(config)
        store.initialize_from_config(config)

        assert store.table_count("models") == 2
        assert store.table_count("prompts") == 2
        assert store.table_count("ratings") == 2
        assert store.model_identifier("model/a") == model_id("model/a")
        assert store.model_identifier("model/b") == model_id("model/b")


def build_config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": tmp_path / "logs" / "completions",
            "log_dir": tmp_path / "logs" / "tournament",
            "state_dir": tmp_path / "logs" / "tournament" / "state",
            "contestant_models": ["model/a", "model/b"],
            "prompts": [
                {"id": "p-1", "text": "Prompt 1"},
                {"id": "p-2", "text": "Prompt 2"},
            ],
            "judge_model": "judge/model",
            "judge_prompt_template": "Prompt: {prompt}",
        }
    )
