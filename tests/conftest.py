"""Conftest for pytest."""

import json
from unittest.mock import Mock

import pytest
from aiohttp import ClientResponse, StreamReader


@pytest.fixture
def load_sample_file_assemblyai() -> dict:
    """Load sample file for AssemblyAI."""
    with open("data/assemblyai_sample.json") as f:
        data = json.load(f)

    return data


@pytest.fixture
def load_sample_file_deepgram() -> dict:
    """Load sample file for AWS."""
    with open("data/deepgram_sample.json") as f:
        data = json.load(f)

    return data


@pytest.fixture
def load_sample_file_revai() -> dict:
    """Load sample file for Rev.ai."""
    with open("data/revai_sample.json") as f:
        data = json.load(f)

    return data


@pytest.fixture
def load_sample_file_speechmatics() -> dict:
    """Load sample file for Speechmatics."""
    with open("data/speechmatics_sample.json") as f:
        data = json.load(f)

    return data


@pytest.fixture
def load_sample_file_wordcab() -> dict:
    """Load sample file for Wordcab."""
    with open("data/wordcab_sample.json") as f:
        data = json.load(f)

    return data


@pytest.fixture
def mock_response() -> Mock:
    """Mock aiohttp.ClientResponse."""
    response = Mock(spec=ClientResponse)
    response.content = Mock(spec=StreamReader)
    response.content.read = Mock(side_effect=[b"mock_content", None])

    return response
