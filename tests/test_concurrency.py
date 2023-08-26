"""Test the concurrency module."""

from typing import Any

import pytest

from rtasr.concurrency import ConcurrencyHandler, ConcurrencyToken


class TestConcurrencyToken:
    """Test the ConcurrencyToken class."""

    @pytest.mark.parametrize("value", [None, "string", True, False, [], {}])
    def test_concurrency_token_invalid(self, value: Any) -> None:
        """Test the concurrency token with invalid values."""
        with pytest.raises(TypeError):
            ConcurrencyToken(value)

    @pytest.mark.parametrize("value", [0, 1, 2, 1.0, 2.0])
    def test_concurrency_token_valid(self, value: int) -> None:
        """Test the concurrency token."""
        token = ConcurrencyToken(value=value)

        assert token.value == value


class TestConcurrencyHandler:
    """Test the ConcurrencyHandler class."""

    @pytest.mark.asyncio
    async def test_concurrency_handler_with_limit(self) -> None:
        """Test the concurrency handler with a limit."""
        limit = 5
        concurrency_handler = ConcurrencyHandler(limit)

        assert hasattr(concurrency_handler, "limit")
        assert hasattr(concurrency_handler, "queue")
        assert concurrency_handler.limit == limit
        assert concurrency_handler.queue.maxsize == limit

        tokens = []
        for _ in range(limit):
            token = await concurrency_handler.get()
            assert isinstance(token, ConcurrencyToken)
            tokens.append(token)

        # Queue should be empty now
        assert concurrency_handler.queue.empty()

        for token in tokens:
            concurrency_handler.put(token)

        # Queue should be refilled
        for _ in range(limit):
            token = await concurrency_handler.get()
            assert isinstance(token, ConcurrencyToken)

    @pytest.mark.asyncio
    async def test_concurrency_handler_without_limit(self) -> None:
        """Test the concurrency handler without a limit."""
        concurrency_handler = ConcurrencyHandler(limit=None)

        token = await concurrency_handler.get()
        assert token is None
