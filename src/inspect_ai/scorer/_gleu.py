from collections import Counter
from collections.abc import Callable

from inspect_ai.solver._task_state import TaskState

from ._metric import Score
from ._metrics import mean, stderr
from ._scorer import Scorer, scorer
from ._target import Target


Tokenize = Callable[[str], list[str]]


@scorer(metrics=[mean(), stderr()])
def gleu(
    answer_fn: Callable[[str], str] | None = None,
    *,
    ignore_case: bool = True,
    min_n: int = 1,
    max_n: int = 4,
    tokenizer: Tokenize | None = None,
) -> Scorer:
    """Scorer which computes a sentence-level GLEU score.

    GLEU is computed as clipped n-gram overlap between the model answer
    and each target reference over the n-gram range [`min_n`, `max_n`].
    For each reference, score is:

        overlap / max(candidate_ngrams, reference_ngrams)

    which is equivalent to `min(precision, recall)` over the n-grams in
    the configured range. The final score is the maximum score over all
    provided target references.

    Args:
       answer_fn: Custom function to extract the answer from the completion
          (defaults to using the completion).
       ignore_case: Do case-insensitive token matching.
       min_n: Minimum n-gram size.
       max_n: Maximum n-gram size.
       tokenizer: Optional custom tokenizer. Defaults to whitespace tokenization.
    """
    _validate_ngram_range(min_n=min_n, max_n=max_n)

    async def score(state: TaskState, target: Target) -> Score:
        answer = (
            answer_fn(state.output.completion) if answer_fn else state.output.completion
        )
        gleu_score = max_gleu_score(
            answer,
            target.target,
            ignore_case=ignore_case,
            min_n=min_n,
            max_n=max_n,
            tokenizer=tokenizer,
        )
        return Score(value=gleu_score, answer=answer)

    return score


def max_gleu_score(
    answer: str,
    targets: list[str],
    *,
    ignore_case: bool = True,
    min_n: int = 1,
    max_n: int = 4,
    tokenizer: Tokenize | None = None,
) -> float:
    """Compute the maximum GLEU score for one answer across references."""
    _validate_ngram_range(min_n=min_n, max_n=max_n)
    answer_tokens = _tokenize(answer, ignore_case=ignore_case, tokenizer=tokenizer)
    return max(
        (
            _compute_gleu(
                answer_tokens,
                _tokenize(target, ignore_case=ignore_case, tokenizer=tokenizer),
                min_n=min_n,
                max_n=max_n,
            )
            for target in targets
        ),
        default=0.0,
    )


def compute_gleu(
    answer: str,
    target: str,
    *,
    ignore_case: bool = True,
    min_n: int = 1,
    max_n: int = 4,
    tokenizer: Tokenize | None = None,
) -> float:
    """Compute GLEU for one answer and one target reference."""
    _validate_ngram_range(min_n=min_n, max_n=max_n)
    answer_tokens = _tokenize(answer, ignore_case=ignore_case, tokenizer=tokenizer)
    target_tokens = _tokenize(target, ignore_case=ignore_case, tokenizer=tokenizer)
    return _compute_gleu(answer_tokens, target_tokens, min_n=min_n, max_n=max_n)


def _compute_gleu(
    answer_tokens: list[str], target_tokens: list[str], *, min_n: int, max_n: int
) -> float:
    answer_ngrams, answer_total = _all_ngrams(answer_tokens, min_n=min_n, max_n=max_n)
    target_ngrams, target_total = _all_ngrams(target_tokens, min_n=min_n, max_n=max_n)

    if answer_total == 0 and target_total == 0:
        return 1.0 if answer_tokens == target_tokens else 0.0
    if answer_total == 0 or target_total == 0:
        return 0.0

    overlap = sum((answer_ngrams & target_ngrams).values())
    return overlap / max(answer_total, target_total)


def _all_ngrams(
    tokens: list[str], *, min_n: int, max_n: int
) -> tuple[Counter[tuple[str, ...]], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    total = 0
    for n in range(min_n, max_n + 1):
        ngrams = _ngram_counts(tokens, n)
        counts.update(ngrams)
        total += sum(ngrams.values())
    return counts, total


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _tokenize(
    text: str, *, ignore_case: bool, tokenizer: Tokenize | None
) -> list[str]:
    tokens = tokenizer(text) if tokenizer else text.split()
    if ignore_case:
        tokens = [token.casefold() for token in tokens]
    return [token for token in tokens if token]


def _validate_ngram_range(*, min_n: int, max_n: int) -> None:
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    if max_n < min_n:
        raise ValueError("max_n must be >= min_n")
