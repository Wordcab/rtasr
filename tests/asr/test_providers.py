"""Test the asr providers."""

from pydantic import HttpUrl, SecretStr

from rtasr.asr.providers import (
    ProviderConfig,
    ProviderResult,
    TranscriptionStatus,
)


class TestProviderConfig:
    """Test the ProviderConfig class."""

    def test_provider_config(self) -> None:
        """Test ProviderConfig."""
        cfg = ProviderConfig(
            api_url="https://example.com",
            api_key="1234567890abcdefghijklmnopqrstuvwxyz",
        )

        assert isinstance(cfg, ProviderConfig)

        assert cfg.api_url == HttpUrl("https://example.com")
        assert str(cfg.api_url) == "https://example.com/"

        assert cfg.api_key == SecretStr("1234567890abcdefghijklmnopqrstuvwxyz")
        assert cfg.api_key.get_secret_value() == "1234567890abcdefghijklmnopqrstuvwxyz"


class TestProviderResult:
    """Test the ProviderResult class."""

    def test_provider_result(self) -> None:
        """Test ProviderResult."""
        result = ProviderResult(
            cached=10,
            completed=20,
            errors=["error1", "error2"],
            failed=30,
            provider_name="test",
        )

        assert isinstance(result, ProviderResult)

        assert result.cached == 10
        assert isinstance(result.cached, int)

        assert result.completed == 20
        assert isinstance(result.completed, int)

        assert result.errors == ["error1", "error2"]
        assert isinstance(result.errors, list)

        assert result.failed == 30
        assert isinstance(result.failed, int)

        assert result.provider_name == "test"
        assert isinstance(result.provider_name, str)


class TestTranscriptionStatus:
    """Test the TranscriptionStatus Enum."""

    def test_transcription_status_enum(self) -> None:
        """Test the transcription status enum."""
        assert len(TranscriptionStatus) == 4

        assert TranscriptionStatus.CACHED == "CACHED"
        assert TranscriptionStatus.COMPLETED == "COMPLETED"
        assert TranscriptionStatus.FAILED == "FAILED"
        assert TranscriptionStatus.IN_PROGRESS == "IN_PROGRESS"
