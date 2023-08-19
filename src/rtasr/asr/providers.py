"""Providers are the classes that actually do the API calls to the different ASR services."""

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List

import aiofiles
import aiohttp
from pydantic import BaseModel, HttpUrl, SecretStr
from rich import print
from rich.live import Live

from rtasr.asr.options import DeepgramOptions
from rtasr.constants import create_live_panel
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

    async def launch(
        self,
        audio_files: List[Path],
        dataset_name: str,
        provider_name: str,
        output_dir: Path,
    ) -> None:
        """Call the API of the ASR provider."""
        url = f"{self.config.api_url}{build_query_string(self.options)}"
        headers = {
            "Authorization": f"Token {self.config.api_key.get_secret_value()}",
        }

        current_progress, step_progress, splits_progress, progress_group = create_live_panel()

        with Live(progress_group):
            async with aiohttp.ClientSession() as session:
                current_progress_task_id = current_progress.add_task(
                    f"Benchmarking on the {dataset_name} dataset"
                )
                splits_progress_task_id = splits_progress.add_task(
                    "", total=len(audio_files)
                )
                tasks: List[Callable] = []
                for audio_file in audio_files:
                    headers["Content-Type"] = f"audio/{audio_file.suffix[1:]}"
                    tasks.append(
                        self._run(
                            audio_file=audio_file, url=url, headers=headers, session=session
                        )
                    )

                results = await asyncio.gather(*tasks)

            for result in results:
                print(result)

    @abstractmethod
    async def _run(
        self, audio_file: Path, url: str, headers: dict, session: aiohttp.ClientSession
    ) -> dict:
        """Run the ASR provider."""
        raise NotImplementedError("The ASR provider must implement the `_run` method.")

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

    async def _run(
        self, audio_file: Path, url: str, headers: dict, session: aiohttp.ClientSession
    ) -> dict:
        """Run the Deepgram ASR provider."""
        async with aiofiles.open(audio_file, mode="rb") as f:
            async with session.post(
                url=url,
                data=f,
                headers=headers,
                raise_for_status=True,
            ) as response:
                content = (await response.text()).strip()

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

    async def _run(
        self, audio_file: Path, url: str, headers: dict, session: aiohttp.ClientSession
    ) -> None:
        """Run the ASR provider."""
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
