from pathlib import Path

from inspect_ai.tournament import TournamentConfig, TournamentStore
from inspect_ai.tournament.scheduler import schedule_match_batch
from inspect_ai.tournament.store import initialize_tournament_store
from inspect_ai.tournament.types import response_id


def test_scheduler_respects_pair_and_prompt_constraints(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    initialize_tournament_store(config)

    with TournamentStore(config.state_dir) as store:
        seed_responses(store, config)

        batch1 = schedule_match_batch(
            config,
            store,
            batch_id="b-1",
            round_index=1,
            seed=123,
            persist=True,
        )
        assert len(batch1.scheduled) == 3
        assert all(match.forced for match in batch1.scheduled)

        first_prompts = {
            canonical_pair(match.model_a, match.model_b): match.prompt_id
            for match in batch1.scheduled
        }

        batch2 = schedule_match_batch(
            config,
            store,
            batch_id="b-2",
            round_index=2,
            seed=123,
            persist=False,
        )
        assert len(batch2.scheduled) == 3
        for match in batch2.scheduled:
            pair = canonical_pair(match.model_a, match.model_b)
            assert match.prompt_id != first_prompts[pair]
            assert match.prompt_uses == 0


def test_scheduler_is_deterministic_for_same_seed_and_state(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    initialize_tournament_store(config)

    with TournamentStore(config.state_dir) as store:
        seed_responses(store, config)
        first = schedule_match_batch(
            config,
            store,
            batch_id="b-det",
            round_index=3,
            seed=77,
            persist=False,
        )
        second = schedule_match_batch(
            config,
            store,
            batch_id="b-det",
            round_index=3,
            seed=77,
            persist=False,
        )

    first_signature = [
        (match.model_a, match.model_b, match.prompt_id) for match in first.scheduled
    ]
    second_signature = [
        (match.model_a, match.model_b, match.prompt_id) for match in second.scheduled
    ]
    assert first_signature == second_signature


def seed_responses(store: TournamentStore, config: TournamentConfig) -> None:
    with store.transaction():
        for model_name in config.contestant_models:
            model_id = store.model_identifier(model_name)
            assert model_id is not None
            for prompt in config.prompts:
                store.upsert_response(
                    response_id=response_id(model_id, prompt.id),
                    model_id=model_id,
                    prompt_id=prompt.id,
                    response_text=f"response for {model_name}:{prompt.id}",
                    source_log="seed.eval",
                    sample_id=prompt.id,
                    sample_uuid=None,
                    commit=False,
                )


def canonical_pair(model_a: str, model_b: str) -> tuple[str, str]:
    return (model_a, model_b) if model_a <= model_b else (model_b, model_a)


def build_config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": tmp_path / "logs" / "completions",
            "log_dir": tmp_path / "logs" / "tournament",
            "state_dir": tmp_path / "logs" / "tournament" / "state",
            "contestant_models": ["model/a", "model/b", "model/c"],
            "prompts": [
                {"id": "p-1", "text": "Prompt 1"},
                {"id": "p-2", "text": "Prompt 2"},
            ],
            "batch_size": 3,
            "min_pair_matches": 1,
            "max_pair_matches": 3,
            "max_prompt_uses_per_pair": 1,
            "judge_model": "judge/model",
            "judge_prompt_template": "Prompt: {prompt}",
        }
    )
