"""Conftest for pytest."""

from unittest.mock import Mock

import pytest
from aiohttp import ClientResponse, StreamReader


@pytest.fixture
def mock_response() -> Mock:
    """Mock aiohttp.ClientResponse."""
    response = Mock(spec=ClientResponse)
    response.content = Mock(spec=StreamReader)
    response.content.read = Mock(side_effect=[b"mock_content", None])

    return response
