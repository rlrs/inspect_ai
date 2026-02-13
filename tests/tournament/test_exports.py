import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import inspect_ai.tournament.orchestrator as orchestrator_module
from inspect_ai.scorer import Score
from inspect_ai.tournament import TournamentConfig, export_rankings, run_tournament
from inspect_ai.tournament.indexer import ResponseIndexReport
from inspect_ai.tournament.types import response_id


def test_export_rankings_writes_json_csv_and_pairwise_matrix(
    tmp_path: Path, monkeypatch: object
) -> None:
    _patch_orchestrator(monkeypatch)
    config = build_config(tmp_path)

    run_tournament(config)
    export_result = export_rankings(config)

    assert export_result.rankings_json.exists()
    assert export_result.rankings_csv.exists()
    assert export_result.pairwise_matrix_csv is not None
    assert export_result.pairwise_matrix_csv.exists()

    rankings_payload = json.loads(export_result.rankings_json.read_text("utf-8"))
    assert rankings_payload["run_status"] == "completed"
    assert len(rankings_payload["models"]) == 3
    assert rankings_payload["models"][0]["rank"] == 1

    with export_result.rankings_csv.open("r", encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 3
    assert set(csv_rows[0].keys()) >= {"rank", "model_id", "model_name", "conservative"}

    with export_result.pairwise_matrix_csv.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        matrix_rows = list(csv.reader(file))
    assert len(matrix_rows) == 4
    assert matrix_rows[0][0:2] == ["model_id", "model_name"]


def test_export_rankings_from_state_dir_target(
    tmp_path: Path, monkeypatch: object
) -> None:
    _patch_orchestrator(monkeypatch)
    config = build_config(tmp_path)

    run_tournament(config)
    export_result = export_rankings(config.state_dir)

    assert export_result.rankings_json.exists()
    assert export_result.rankings_csv.exists()


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
