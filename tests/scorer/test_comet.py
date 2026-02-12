import pytest
from test_helpers.utils import simple_task_state

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.scorer import Target, comet


@pytest.mark.anyio
async def test_comet_uses_best_target_reference(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], list[str], list[str] | None]] = []

    class FakeRuntime:
        pass

    def fake_get_runtime(*_args, **_kwargs):
        return FakeRuntime()

    def fake_score(
        _runtime: FakeRuntime,
        sources: list[str],
        candidates: list[str],
        references: list[str] | None,
    ) -> list[float]:
        calls.append((sources, candidates, references))
        assert references is not None
        return [0.1, 0.7, 0.3]

    monkeypatch.setattr("inspect_ai.scorer._comet._get_comet_runtime", fake_get_runtime)
    monkeypatch.setattr("inspect_ai.scorer._comet.score_comet_texts", fake_score)

    scorer = comet(source=lambda _state: "source text")
    state = simple_task_state(model_output="candidate text")
    result = await scorer(state, Target(["ref-1", "ref-2", "ref-3"]))

    assert result.as_float() == pytest.approx(0.7)
    assert result.answer == "candidate text"
    assert result.metadata is not None
    assert result.metadata["reference"] == "ref-2"
    assert calls == [
        (
            ["source text", "source text", "source text"],
            ["candidate text", "candidate text", "candidate text"],
            ["ref-1", "ref-2", "ref-3"],
        )
    ]


@pytest.mark.anyio
async def test_comet_reads_source_and_reference_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeRuntime:
        pass

    def fake_get_runtime(*_args, **_kwargs):
        return FakeRuntime()

    captured: dict[str, list[str] | None] = {}

    def fake_score(
        _runtime: FakeRuntime,
        sources: list[str],
        candidates: list[str],
        references: list[str] | None,
    ) -> list[float]:
        captured["sources"] = sources
        captured["candidates"] = candidates
        captured["references"] = references
        return [0.42]

    monkeypatch.setattr("inspect_ai.scorer._comet._get_comet_runtime", fake_get_runtime)
    monkeypatch.setattr("inspect_ai.scorer._comet.score_comet_texts", fake_score)

    scorer = comet(source="src", reference="ref")
    state = simple_task_state(model_output="candidate text")
    state.metadata = {"src": "source text", "ref": "reference text"}

    result = await scorer(state, Target(["unused-target"]))

    assert result.as_float() == pytest.approx(0.42)
    assert captured == {
        "sources": ["source text"],
        "candidates": ["candidate text"],
        "references": ["reference text"],
    }


@pytest.mark.anyio
async def test_comet_missing_metadata_source_raises_prerequisite_error():
    scorer = comet(source="src")
    state = simple_task_state(model_output="candidate text")

    with pytest.raises(
        PrerequisiteError, match=r"metadata\['src'\]"
    ):
        await scorer(state, Target(["reference text"]))
