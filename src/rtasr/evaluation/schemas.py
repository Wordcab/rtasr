"""Schemas for evaluation module."""

from enum import Enum
from typing import List

from pydantic import BaseModel


class ProviderResult(BaseModel):
    """The evaluation result for a provider."""

    cached: int
    evaluated: int
    not_found: int
    provider_name: str


class EvaluationResult(BaseModel):
    """The DER evaluation result."""

    errors: List[str]
    split_name: str
    results: List[ProviderResult]


class EvaluationStatus(str, Enum):
    """Status of the evaluation."""

    CACHED = "CACHED"
    EVALUATED = "EVALUATED"
    NOT_FOUND = "NOT_FOUND"
