from inspect_ai.tournament.scorer import (
    parse_judge_decision,
    reconcile_side_swap,
    render_judge_prompt,
)


def test_parse_judge_decision_accepts_single_terminal_decision_line() -> None:
    parsed = parse_judge_decision(
        "Comparison notes.\nMore notes.\nDECISION: b"
    )
    assert parsed.valid is True
    assert parsed.decision == "B"
    assert parsed.parse_error is None


def test_parse_judge_decision_requires_terminal_decision_line() -> None:
    parsed = parse_judge_decision(
        "DECISION: A\nTrailing line makes this invalid."
    )
    assert parsed.valid is False
    assert parsed.decision == "INVALID"
    assert parsed.parse_error == "missing_terminal_decision"


def test_parse_judge_decision_rejects_multiple_decision_lines() -> None:
    parsed = parse_judge_decision(
        "First:\nDECISION: A\nSecond:\nDECISION: A"
    )
    assert parsed.valid is False
    assert parsed.decision == "INVALID"
    assert parsed.parse_error == "multiple_decision_lines"


def test_reconcile_side_swap_resolves_consistency_and_disagreement() -> None:
    # In side=ba, B means canonical A.
    assert reconcile_side_swap("A", "B", invalid_policy="skip") == "A"
    assert reconcile_side_swap("A", "A", invalid_policy="skip") == "INVALID"
    assert reconcile_side_swap("A", "A", invalid_policy="count_as_tie") == "TIE"


def test_render_judge_prompt_fills_expected_variables() -> None:
    prompt = render_judge_prompt(
        template="Prompt: {prompt}\nA:{response_a}\nB:{response_b}",
        prompt="Write a story.",
        response_a="Story A",
        response_b="Story B",
    )
    assert "Prompt: Write a story." in prompt
    assert "A:Story A" in prompt
    assert "B:Story B" in prompt
