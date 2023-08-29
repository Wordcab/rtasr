"""Schemas for evaluation module."""

from enum import Enum
from typing import List

from pydantic import BaseModel, NonNegativeInt


class EvaluationStatus(str, Enum):
    """Status of the evaluation."""

    CACHED = "CACHED"
    EVALUATED = "EVALUATED"
    NOT_FOUND = "NOT_FOUND"


class TaskStatus(str, Enum):
    """Status of an evaluation task."""

    DONE = "DONE"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"


class ProviderResult(BaseModel):
    """The evaluation result for a provider."""

    cached: NonNegativeInt
    evaluated: NonNegativeInt
    not_found: NonNegativeInt
    provider_name: str


class EvaluationResult(BaseModel):
    """The DER evaluation result."""

    errors: List[str]
    split_name: str
    results: List[ProviderResult]
