from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import inspect_ai.tournament.orchestrator as orchestrator_module
from inspect_ai.scorer import Score
from inspect_ai.tournament import (
    TournamentConfig,
    resume_tournament,
    run_tournament,
    tournament_status,
)
from inspect_ai.tournament.indexer import ResponseIndexReport
from inspect_ai.tournament.types import response_id


def test_resume_matches_uninterrupted_final_rankings(
    tmp_path: Path, monkeypatch: object
) -> None:
    _patch_orchestrator(monkeypatch)

    config_partial = build_config(tmp_path / "partial")
    partial = run_tournament(config_partial, max_batches=1)
    assert partial.batches_completed == 1

    resumed = resume_tournament(config_partial)

    config_full = build_config(tmp_path / "full")
    full = run_tournament(config_full)

    assert ranking_signature(resumed.status.standings) == ranking_signature(
        full.status.standings
    )
    assert resumed.status.total_matches == full.status.total_matches
    assert resumed.status.rated_matches == full.status.rated_matches
    assert resumed.status.stop_reasons == full.status.stop_reasons
    assert resumed.status.pending_batch_id is None


def test_status_and_resume_work_from_state_dir(
    tmp_path: Path, monkeypatch: object
) -> None:
    _patch_orchestrator(monkeypatch)
    config = build_config(tmp_path / "status")

    run_tournament(config, max_batches=1)
    mid_status = tournament_status(config.state_dir)

    assert mid_status.run_status == "running"
    assert mid_status.rated_matches > 0
    assert mid_status.pending_batch_id is None
    assert {standing.model_name for standing in mid_status.standings} == set(
        config.contestant_models
    )

    resume_tournament(config.state_dir)
    final_status = tournament_status(config.state_dir)
    assert final_status.run_status == "completed"
    assert len(final_status.stop_reasons) > 0
    assert {standing.model_name for standing in final_status.standings} == set(
        config.contestant_models
    )


def _patch_orchestrator(monkeypatch: object) -> None:
    monkeypatch.setattr(orchestrator_module, "run_generation", _fake_run_generation)
    monkeypatch.setattr(
        orchestrator_module,
        "index_generation_responses",
        _fake_index_generation_responses,
    )
    monkeypatch.setattr(orchestrator_module, "run_judge_batch", _fake_run_judge_batch)


def _fake_run_generation(
    config: TournamentConfig,
    *,
    models: Sequence[str] | None = None,
) -> None:
    del config, models


def _fake_index_generation_responses(
    config: TournamentConfig,
    *,
    store: object | None = None,
) -> ResponseIndexReport:
    assert store is not None
    with store.transaction():
        for model_name in config.contestant_models:
            model_identifier = store.model_identifier(model_name)
            assert model_identifier is not None
            for prompt in config.prompts:
                store.upsert_response(
                    response_id=response_id(model_identifier, prompt.id),
                    model_id=model_identifier,
                    prompt_id=prompt.id,
                    response_text=f"{model_name}:{prompt.id}",
                    source_log="synthetic.eval",
                    sample_id=prompt.id,
                    sample_uuid=None,
                    commit=False,
                )

    return ResponseIndexReport(
        logs_seen=0,
        logs_processed=0,
        samples_seen=0,
        responses_indexed=len(config.contestant_models) * len(config.prompts),
        responses_inserted=len(config.contestant_models) * len(config.prompts),
        skipped_samples=0,
        log_errors=0,
        missing_by_model={model_name: [] for model_name in config.contestant_models},
    )


def _fake_run_judge_batch(
    config: TournamentConfig,
    matches: Sequence[object],
    *,
    grader_model: object | None = None,
    log_dir: str | Path | None = None,
) -> object:
    del grader_model
    samples: list[object] = []
    for match in matches:
        winner = min(match.model_a, match.model_b)
        decision_ab = "A" if winner == match.model_a else "B"
        score_ab = Score(
            value=decision_ab,
            explanation=f"DECISION: {decision_ab}",
            metadata={"judge_model": "mock_judge"},
        )
        samples.append(
            SimpleNamespace(
                id=f"{match.match_id}:ab",
                metadata={"match_id": match.match_id, "side": "ab"},
                scores={"pairwise_judge": score_ab},
                uuid=None,
            )
        )

        if config.side_swap:
            decision_ba = "A" if winner == match.model_b else "B"
            score_ba = Score(
                value=decision_ba,
                explanation=f"DECISION: {decision_ba}",
                metadata={"judge_model": "mock_judge"},
            )
            samples.append(
                SimpleNamespace(
                    id=f"{match.match_id}:ba",
                    metadata={"match_id": match.match_id, "side": "ba"},
                    scores={"pairwise_judge": score_ba},
                    uuid=None,
                )
            )

    fake_log = SimpleNamespace(
        samples=samples,
        location=(Path(log_dir) / "synthetic.eval").as_posix()
        if log_dir is not None
        else "synthetic.eval",
    )
    return SimpleNamespace(logs=[fake_log])


def ranking_signature(standings: Sequence[object]) -> list[tuple[str, int, float, float]]:
    return [
        (
            standing.model_id,
            standing.games,
            round(standing.mu, 6),
            round(standing.sigma, 6),
        )
        for standing in standings
    ]


def build_config(base_dir: Path) -> TournamentConfig:
    return TournamentConfig.model_validate(
        {
            "completion_log_dir": base_dir / "logs" / "completions",
            "log_dir": base_dir / "logs" / "tournament",
            "state_dir": base_dir / "logs" / "tournament" / "state",
            "contestant_models": ["model/a", "model/b", "model/c"],
            "prompts": [
                {"id": "p-1", "text": "Prompt 1"},
                {"id": "p-2", "text": "Prompt 2"},
            ],
            "batch_size": 2,
            "max_total_matches": 6,
            "min_pair_matches": 1,
            "max_pair_matches": 8,
            "max_prompt_uses_per_pair": 4,
            "p_stop": 0.9999,
            "epsilon": 10.0,
            "n_stable_batches": 8,
            "seed": 11,
            "judge_model": "judge/model",
            "judge_max_samples": 8,
            "judge_prompt_template": "Prompt: {prompt}",
            "side_swap": True,
        }
    )
