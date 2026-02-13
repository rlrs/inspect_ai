from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from inspect_ai.log import (
    EvalSample,
    list_eval_logs,
    read_eval_log,
    read_eval_log_samples,
)

from .config import TournamentConfig, load_tournament_config
from .store import TournamentStore
from .types import response_id


class ResponseIndexReport(BaseModel):
    """Summary of a response indexing run."""

    logs_seen: int = 0
    logs_processed: int = 0
    samples_seen: int = 0
    responses_indexed: int = 0
    responses_inserted: int = 0
    skipped_samples: int = 0
    log_errors: int = 0
    missing_by_model: dict[str, list[str]]

    @property
    def missing_count(self) -> int:
        """Total number of missing model/prompt responses."""
        return sum(len(prompt_ids) for prompt_ids in self.missing_by_model.values())


def index_generation_responses(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
    *,
    store: TournamentStore | None = None,
) -> ResponseIndexReport:
    """Index generation logs into tournament response state."""
    parsed = load_tournament_config(config)
    if parsed.state_dir is None:
        raise ValueError("state_dir is required")

    if store is not None:
        return _index_generation_responses(parsed, store)

    with TournamentStore(parsed.state_dir) as opened_store:
        return _index_generation_responses(parsed, opened_store)


def _index_generation_responses(
    config: TournamentConfig,
    store: TournamentStore,
) -> ResponseIndexReport:
    store.initialize_from_config(config)

    expected_models = set(config.contestant_models)
    expected_prompt_ids = [prompt.id for prompt in config.prompts]
    expected_prompt_set = set(expected_prompt_ids)
    report = ResponseIndexReport(missing_by_model={})

    with store.transaction():
        for log_info in list_eval_logs(config.completion_log_dir.as_posix()):
            report.logs_seen += 1
            try:
                header = read_eval_log(log_info, header_only=True)
                model_name = header.eval.model
                if model_name not in expected_models:
                    continue
                model_identifier = store.model_identifier(model_name)
                if model_identifier is None:
                    continue

                report.logs_processed += 1
                for sample in read_eval_log_samples(
                    log_info,
                    all_samples_required=False,
                ):
                    report.samples_seen += 1
                    prompt_id = _resolve_prompt_id(sample, config.prompt_id_field)
                    if prompt_id is None or prompt_id not in expected_prompt_set:
                        report.skipped_samples += 1
                        continue

                    inserted = store.upsert_response(
                        response_id=response_id(model_identifier, prompt_id),
                        model_id=model_identifier,
                        prompt_id=prompt_id,
                        response_text=sample.output.completion,
                        source_log=_relative_log_name(
                            config.completion_log_dir, log_info.name
                        ),
                        sample_id=str(sample.id),
                        sample_uuid=sample.uuid,
                        commit=False,
                    )
                    report.responses_indexed += 1
                    if inserted:
                        report.responses_inserted += 1
            except Exception:
                report.log_errors += 1

    report.missing_by_model = store.missing_prompt_ids_by_model(
        config.contestant_models, expected_prompt_ids
    )
    return report


def _resolve_prompt_id(sample: EvalSample, prompt_id_field: str) -> str | None:
    metadata = sample.metadata if sample.metadata is not None else {}
    prompt_value = metadata.get(prompt_id_field)
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, int):
        return str(prompt_value)
    if isinstance(sample.id, str):
        return sample.id
    if isinstance(sample.id, int):
        return str(sample.id)
    return None


def _relative_log_name(base_dir: Path, log_name: str) -> str:
    if "://" in log_name:
        return log_name

    try:
        return Path(log_name).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return log_name
