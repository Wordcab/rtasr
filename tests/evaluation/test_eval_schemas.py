"""Test the schemas of the evaluation module."""

from typing import List

import pytest

from rtasr.evaluation.schemas import (
    EvaluationResult,
    EvaluationStatus,
    ProviderResult,
    TaskStatus,
)


class TestSchemasEnums:
    """Test the enums of the schemas."""

    def test_evaluation_status(self) -> None:
        """Test the EvaluationStatus enum."""
        assert EvaluationStatus.CACHED == "CACHED"
        assert EvaluationStatus.EVALUATED == "EVALUATED"
        assert EvaluationStatus.NOT_FOUND == "NOT_FOUND"

    @pytest.mark.parametrize(
        "status",
        [
            "CACHED",
            "EVALUATED",
            "NOT_FOUND",
            "cached",
            "evaluated",
            "not_found",
        ],
    )
    def test_evaluation_status_str(self, status: str) -> None:
        """Test the EvaluationStatus enum with string."""
        assert EvaluationStatus[status.upper()] == EvaluationStatus(status.upper())

    def test_task_status(self) -> None:
        """Test the TaskStatus enum."""
        assert TaskStatus.DONE == "DONE"
        assert TaskStatus.ERROR == "ERROR"
        assert TaskStatus.IN_PROGRESS == "IN_PROGRESS"

    @pytest.mark.parametrize(
        "status",
        [
            "DONE",
            "ERROR",
            "IN_PROGRESS",
            "done",
            "error",
            "in_progress",
        ],
    )
    def test_task_status_str(self, status: str) -> None:
        """Test the TaskStatus enum with string."""
        assert TaskStatus[status.upper()] == TaskStatus(status.upper())


class TestSchemasBaseModel:
    """Test the BaseModel of the schemas."""

    @pytest.mark.parametrize(
        "cached, evaluated, not_found, provider_name",
        [
            (0, 0, 0, "test"),
            (1, 2, 3, "test"),
            (127, 0, 3, "bro_asr_provider_3"),
        ],
    )
    def test_provider_result_valid(
        self, cached: int, evaluated: int, not_found: int, provider_name: str
    ) -> None:
        """Test the ProviderResult model."""
        provider_result = ProviderResult(
            cached=cached,
            evaluated=evaluated,
            not_found=not_found,
            provider_name=provider_name,
        )
        assert provider_result.cached == cached
        assert isinstance(provider_result.cached, int)
        assert provider_result.evaluated == evaluated
        assert isinstance(provider_result.evaluated, int)
        assert provider_result.not_found == not_found
        assert isinstance(provider_result.not_found, int)
        assert provider_result.provider_name == provider_name
        assert isinstance(provider_result.provider_name, str)

    @pytest.mark.parametrize(
        "cached, evaluated, not_found, provider_name",
        [
            (-1, 0, 0, "test"),
            (0, -1, 0, "test"),
            (0, 0, -1, "test"),
            (0, 0, 0, 1),
        ],
    )
    def test_provider_result_invalid(
        self, cached: int, evaluated: int, not_found: int, provider_name: str
    ) -> None:
        """Test the ProviderResult model."""
        with pytest.raises(ValueError):
            ProviderResult(
                cached=cached,
                evaluated=evaluated,
                not_found=not_found,
                provider_name=provider_name,
            )

    @pytest.mark.parametrize(
        "errors, split_name, results",
        [
            ([], "test", []),
            (["test"], "test", []),
            (
                [],
                "test",
                [
                    ProviderResult(
                        cached=0, evaluated=0, not_found=0, provider_name="test"
                    )
                ],
            ),
            (
                ["test"],
                "test",
                [
                    ProviderResult(
                        cached=0, evaluated=0, not_found=0, provider_name="test"
                    )
                ],
            ),
            (
                ["test"],
                "test",
                [
                    ProviderResult(
                        cached=0, evaluated=0, not_found=0, provider_name="test"
                    ),
                    ProviderResult(
                        cached=0, evaluated=0, not_found=0, provider_name="test"
                    ),
                ],
            ),
        ],
    )
    def test_evaluation_result_valid(
        self, errors: List[str], split_name: str, results: List[ProviderResult]
    ) -> None:
        """Test the EvaluationResult model."""
        evaluation_result = EvaluationResult(
            errors=errors, split_name=split_name, results=results
        )
        assert evaluation_result.errors == errors
        assert isinstance(evaluation_result.errors, list)
        assert evaluation_result.split_name == split_name
        assert isinstance(evaluation_result.split_name, str)
        assert evaluation_result.results == results
        assert isinstance(evaluation_result.results, list)

    @pytest.mark.parametrize(
        "errors, split_name, results",
        [
            (["test"], 0, []),
            (
                "test",
                "test",
                [
                    ProviderResult(
                        cached=0, evaluated=0, not_found=0, provider_name="test"
                    )
                ],
            ),
            ([], "test", [1, 2, 3]),
        ],
    )
    def test_evaluation_result_invalid(
        self, errors: List[str], split_name: str, results: List[ProviderResult]
    ) -> None:
        """Test the EvaluationResult model."""
        with pytest.raises(ValueError):
            EvaluationResult(errors=errors, split_name=split_name, results=results)
