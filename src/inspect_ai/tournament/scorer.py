import re
from typing import Any, Literal

from pydantic import BaseModel, JsonValue

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, metric, scorer
from inspect_ai.solver import TaskState

from .types import Decision, InvalidPolicy

DECISION_PATTERN = re.compile(r"^DECISION:\s*(A|B|TIE|INVALID)\s*$", re.IGNORECASE)


class ParsedJudgeDecision(BaseModel):
    """Parsed decision from judge output."""

    decision: Decision
    valid: bool
    parse_error: str | None = None
    decision_line: str | None = None


@metric
def decision_valid_rate() -> Metric:
    """Rate of judge scores with non-INVALID decisions."""

    def metric_fn(scores: list[SampleScore]) -> float:
        if len(scores) == 0:
            return 0.0
        valid = sum(1 for sample_score in scores if sample_score.score.value != "INVALID")
        return valid / len(scores)

    return metric_fn


@scorer(metrics=[decision_valid_rate()], name="pairwise_judge")
def pairwise_judge(
    prompt_template: str,
    model: str | Model | None = None,
    model_role: str | None = "grader",
    generate_config: GenerateConfig | dict[str, JsonValue] | None = None,
) -> Scorer:
    """Judge a pairwise comparison and return one of A/B/TIE/INVALID."""
    model_spec = model
    resolved_model: Model | None = model if isinstance(model, Model) else None
    resolved_config = _resolve_generate_config(generate_config)

    async def score(state: TaskState, target: Any) -> Score:
        del target
        metadata = state.metadata if state.metadata is not None else {}
        prompt = str(metadata.get("prompt", state.input_text))
        response_a = metadata.get("response_a")
        response_b = metadata.get("response_b")
        if not isinstance(response_a, str) or not isinstance(response_b, str):
            return Score(
                value="INVALID",
                explanation="Missing required response_a/response_b metadata for judge scoring.",
                metadata={"parse_error": "missing_responses", "side": metadata.get("side")},
            )

        judge_prompt = render_judge_prompt(
            template=prompt_template,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
        )

        nonlocal resolved_model
        if resolved_model is None:
            if model_spec is not None:
                resolved_model = (
                    model_spec if isinstance(model_spec, Model) else get_model(model_spec)
                )
            elif model_role is not None:
                resolved_model = get_model(role=model_role)
            else:
                resolved_model = get_model()

        judge_output = await resolved_model.generate(judge_prompt, config=resolved_config)
        parsed = parse_judge_decision(judge_output.completion)
        return Score(
            value=parsed.decision,
            explanation=judge_output.completion,
            metadata={
                "parse_valid": parsed.valid,
                "parse_error": parsed.parse_error,
                "decision_line": parsed.decision_line,
                "judge_prompt": judge_prompt,
                "judge_model": str(resolved_model),
                "side": metadata.get("side"),
                "match_id": metadata.get("match_id"),
            },
        )

    return score


def parse_judge_decision(text: str) -> ParsedJudgeDecision:
    """Strictly parse judge completion into A/B/TIE/INVALID."""
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip() != ""]
    if len(non_empty_lines) == 0:
        return ParsedJudgeDecision(
            decision="INVALID", valid=False, parse_error="empty_completion"
        )

    terminal_line = non_empty_lines[-1]
    terminal_match = DECISION_PATTERN.match(terminal_line)
    if terminal_match is None:
        return ParsedJudgeDecision(
            decision="INVALID",
            valid=False,
            parse_error="missing_terminal_decision",
        )

    decision_lines = [line for line in non_empty_lines if DECISION_PATTERN.match(line)]
    if len(decision_lines) != 1:
        return ParsedJudgeDecision(
            decision="INVALID",
            valid=False,
            parse_error="multiple_decision_lines",
        )

    decision_str = terminal_match.group(1).upper()
    if decision_str == "A":
        decision: Decision = "A"
    elif decision_str == "B":
        decision = "B"
    elif decision_str == "TIE":
        decision = "TIE"
    else:
        decision = "INVALID"
    return ParsedJudgeDecision(
        decision=decision, valid=True, parse_error=None, decision_line=terminal_line
    )


def canonicalize_side_decision(
    decision: Decision, side: Literal["ab", "ba"]
) -> Decision:
    """Map side-specific decisions into canonical (ab) space."""
    if side == "ab":
        return decision

    if decision == "A":
        return "B"
    if decision == "B":
        return "A"
    return decision


def reconcile_side_swap(
    decision_ab: Decision,
    decision_ba: Decision,
    *,
    invalid_policy: InvalidPolicy = "skip",
) -> Decision:
    """Reconcile side-swapped judgments into one canonical decision."""
    canonical_ab = canonicalize_side_decision(decision_ab, "ab")
    canonical_ba = canonicalize_side_decision(decision_ba, "ba")

    if "INVALID" in (canonical_ab, canonical_ba):
        return "INVALID"

    if canonical_ab == canonical_ba:
        return canonical_ab

    if invalid_policy == "count_as_tie":
        return "TIE"
    return "INVALID"


def render_judge_prompt(
    *,
    template: str,
    prompt: str,
    response_a: str,
    response_b: str,
) -> str:
    """Render the judge prompt from template variables."""
    return template.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )


def _resolve_generate_config(
    generate_config: GenerateConfig | dict[str, JsonValue] | None,
) -> GenerateConfig:
    if isinstance(generate_config, GenerateConfig):
        return generate_config
    if isinstance(generate_config, dict):
        return GenerateConfig.model_validate(generate_config)
    return GenerateConfig()
