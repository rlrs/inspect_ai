import hashlib
from typing import Literal, Sequence

from pydantic import BaseModel, Field

Decision = Literal["A", "B", "TIE", "INVALID"]
CanonicalDecision = Literal["A", "B", "TIE"]
InvalidPolicy = Literal["skip", "count_as_tie"]


def deterministic_id(namespace: str, *parts: str, length: int = 16) -> str:
    """Create a deterministic identifier from a namespace and string parts."""
    if namespace.strip() == "":
        raise ValueError("namespace must not be empty")
    if length <= 0:
        raise ValueError("length must be greater than 0")
    if len(parts) == 0:
        raise ValueError("at least one part is required")

    payload = "\x1f".join((namespace, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}_{digest[:length]}"


def model_id(model_name: str) -> str:
    """Create a deterministic model identifier."""
    return deterministic_id("model", model_name, length=20)


def response_id(model_identifier: str, prompt_id: str) -> str:
    """Create a deterministic response identifier."""
    return deterministic_id("response", model_identifier, prompt_id, length=20)


def match_id(
    model_a: str,
    model_b: str,
    prompt_id: str,
    round_index: int,
    batch_id: str,
) -> str:
    """Create a deterministic match identifier."""
    return deterministic_id(
        "match",
        model_a,
        model_b,
        prompt_id,
        str(round_index),
        batch_id,
        length=20,
    )


def default_project_id(
    models: Sequence[str], prompt_ids: Sequence[str], seed: int | None = None
) -> str:
    """Create a stable project identifier from config-defining fields."""
    model_part = ",".join(sorted(models))
    prompt_part = ",".join(sorted(prompt_ids))
    seed_part = "" if seed is None else str(seed)
    return deterministic_id("project", model_part, prompt_part, seed_part, length=20)


class ModelRating(BaseModel):
    """Current tournament rating for one model."""

    model_id: str
    mu: float
    sigma: float
    games: int = Field(default=0, ge=0)
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    ties: int = Field(default=0, ge=0)

    def conservative_score(self, conservative_k: float) -> float:
        """Compute conservative ranking score."""
        return self.mu - (conservative_k * self.sigma)
