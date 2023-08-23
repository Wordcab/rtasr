"""Test constants.py module."""

from typing import OrderedDict

import pytest

from rtasr.constants import DATASETS, PROVIDERS, Metrics


class TestConstants:
    """Test the DATASETS, PROVIDERS, and Metrics constants."""

    def test_datasets(self) -> None:
        """Test DATASETS constant."""
        assert isinstance(DATASETS, OrderedDict)
        # TODO: Add more tests.

    def test_providers(self) -> None:
        """Test PROVIDERS constant."""
        assert isinstance(PROVIDERS, OrderedDict)
        # TODO: Add more tests.

    def test_metrics(self) -> None:
        """Test Metrics enum."""
        assert Metrics.__members__ == {
            "DER": "DER",
            "WER": "WER",
        }

        assert Metrics.DER == "DER"
        assert Metrics.WER == "WER"

        assert Metrics.DER == Metrics("DER")
        assert Metrics.WER == Metrics("WER")

        with pytest.raises(ValueError):
            Metrics("der")
        with pytest.raises(ValueError):
            Metrics("wer")
