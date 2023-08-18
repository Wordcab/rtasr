"""Providers are the classes that actually do the API calls to the different ASR services."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import aiohttp
from pydantic import BaseModel, HttpUrl, SecretStr

from rtasr.asr.options import DeepgramOptions
from rtasr.utils import build_query_string


class ProviderConfig(BaseModel):
    """The base class for all ASR provider configurations."""

    api_url: HttpUrl
    api_key: SecretStr


class ASRProvider(ABC):
    """The base class for all ASR providers."""

    def __init__(self, api_url: str, api_key: str) -> None:
        """Initialize the ASR provider."""
        self.config = ProviderConfig(api_url=api_url, api_key=api_key)

    @abstractmethod
    async def api_call(self) -> None:
        """Call the API of the ASR provider."""
        raise NotImplementedError(
            "The ASR provider must implement the `api_call` method."
        )

    @abstractmethod
    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        raise NotImplementedError(
            "The ASR provider must implement the `result_to_rttm` method."
        )


class AssemblyAI(ASRProvider):
    """The ASR provider class for AssemblyAI."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the AssemblyAI ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Aws(ASRProvider):
    """The ASR provider class for AWS."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the AWS ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Azure(ASRProvider):
    """The ASR provider class for Azure."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the Azure ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Deepgram(ASRProvider):
    """The ASR provider class for Deepgram."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        """Initialize the Deepgram ASR provider."""
        super().__init__(api_url, api_key)
        self.options = DeepgramOptions(**options)

    async def api_call(self, audio_file: Path) -> None:
        """Call the API of the Deepgram ASR provider."""
        audio: bytes = open(audio_file, "rb").encode("utf-8")

        async with aiohttp.request(
            method="POST",
            url=f"{self.config.api_url}{build_query_string(self.options)}",
            data=audio,
            headers={
                "Content-Type": f"audio/{audio_file.suffix[1:]}",
                "Authorization": f"Token {self.config.api_key.get_secret_value()}",
            },
            raise_for_status=True,
        ) as resp:
            content = (await resp.text()).strip()

            if not content:
                return None

            body = json.loads(content)

            if body.get("error"):
                raise Exception(f"DG: {content}")

            return body

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Google(ASRProvider):
    """The ASR provider class for Google."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the Google ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class RevAI(ASRProvider):
    """The ASR provider class for RevAI."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the RevAI ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Speechmatics(ASRProvider):
    """The ASR provider class for Speechmatics."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the Speechmatics ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Wordcab(ASRProvider):
    """The ASR provider class for Wordcab."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        """Initialize the Wordcab ASR provider."""
        super().__init__(api_url, api_key)
        self.options = options

    async def api_call(self) -> None:
        """Call the API of the Wordcab ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass
