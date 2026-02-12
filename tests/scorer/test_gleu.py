import pytest
from test_helpers.utils import simple_task_state

from inspect_ai.scorer import Target, gleu
from inspect_ai.scorer._gleu import compute_gleu


@pytest.mark.anyio
async def test_gleu_exact_match():
    scorer = gleu()
    state = simple_task_state(model_output="the cat is on the mat")
    result = await scorer(state, Target(["the cat is on the mat"]))

    assert result.as_float() == 1.0


@pytest.mark.anyio
async def test_gleu_partial_match():
    scorer = gleu()
    state = simple_task_state(model_output="the cat is on the mat")
    result = await scorer(state, Target(["the cat sat on the mat"]))

    assert result.as_float() == pytest.approx(0.5)


@pytest.mark.anyio
async def test_gleu_uses_best_target():
    scorer = gleu()
    state = simple_task_state(model_output="the cat is on the mat")
    result = await scorer(
        state,
        Target(
            [
                "completely unrelated words",
                "the cat sat on the mat",
            ]
        ),
    )

    assert result.as_float() == pytest.approx(0.5)


@pytest.mark.anyio
async def test_gleu_ignore_case_false():
    scorer = gleu(ignore_case=False)
    state = simple_task_state(model_output="Hello world")
    result = await scorer(state, Target(["hello world"]))

    assert result.as_float() == pytest.approx(1 / 3)


def test_gleu_invalid_ngram_range():
    with pytest.raises(ValueError, match="min_n must be >= 1"):
        gleu(min_n=0)

    with pytest.raises(ValueError, match="max_n must be >= min_n"):
        gleu(min_n=3, max_n=2)


def test_compute_gleu_empty_text():
    assert compute_gleu("", "") == 1.0
    assert compute_gleu("", "hello") == 0.0
