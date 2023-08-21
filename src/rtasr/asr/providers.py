"""Providers are the classes that actually do the API calls to the different ASR services."""

import asyncio
import json
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List

import aiofiles
import aiohttp
from pydantic import BaseModel, HttpUrl, SecretStr
from rich import print
from rich.progress import Progress, TaskID

from rtasr.asr.options import (
    AssemblyAIOptions,
    AwsOptions,
    AzureOptions,
    DeepgramOptions,
    GoogleOptions,
    RevAIOptions,
    SpeechmaticsOptions,
    WordcabOptions,
)
from rtasr.utils import build_query_string


class ProviderConfig(BaseModel):
    """The base class for all ASR provider configurations."""

    api_url: HttpUrl
    api_key: SecretStr


class ProviderResult(BaseModel):
    """The base class for all ASR provider results."""

    provider_name: str
    completed: int
    failed: int


class TranscriptionStatus(str, Enum):
    """Status of the transcription."""

    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ASRProvider(ABC):
    """The base class for all ASR providers."""

    def __init__(self, api_url: str, api_key: str) -> None:
        """Initialize the ASR provider."""
        self.config = ProviderConfig(api_url=api_url, api_key=api_key)

    async def launch(
        self,
        audio_files: Dict[str, List[Path]],
        output_dir: Path,
        session: aiohttp.ClientSession,
        split_progress: Progress,
        split_progress_task_id: TaskID,
        step_progress: Progress,
    ) -> None:
        """Call the API of the ASR provider."""
        url = f"{self.config.api_url}{build_query_string(self.options)}"
        headers = {
            "Authorization": f"Token {self.config.api_key.get_secret_value()}",
        }

        step_progress_task_id = step_progress.add_task(
            "",
            action=f"[bold green]{self.__class__.__name__}[/bold green]",
            total=len(audio_files),
        )

        task_tracking: Dict[str, Any] = {}

        test_counter = 0  # TODO: Remove this line, test purpose only
        for split_name, split_audio_files in audio_files.items():
            tasks: List[Callable] = []
            for audio_file in split_audio_files:
                if test_counter < 2:  # TODO: Remove this line, test purpose only
                    headers["Content-Type"] = f"audio/{audio_file.suffix[1:]}"
                    tasks.append(
                        self._launch(
                            audio_file=audio_file,
                            url=url,
                            headers=headers,
                            session=session,
                        )
                    )
                    task_tracking[audio_file.name] = {
                        "status": TranscriptionStatus.IN_PROGRESS,
                        "audio_file_name": audio_file.name,
                        "split": split_name,
                    }
                    test_counter += 1  # TODO: Remove this line, test purpose only

            for future in asyncio.as_completed(tasks):
                try:
                    task_result = await future
                    audio_file_name, status, body = task_result
                    if status == TranscriptionStatus.COMPLETED:
                        task_tracking[audio_file_name]["status"] = status
                        _split = task_tracking[audio_file_name]["split"]
                        await self._save_result(
                            audio_file_name=audio_file_name,
                            asr_output=body,
                            output_dir=output_dir / _split,
                        )
                    elif status == TranscriptionStatus.FAILED:
                        task_tracking[audio_file_name]["status"] = status
                except Exception as e:
                    raise Exception(e) from e
                finally:
                    step_progress.advance(step_progress_task_id)

            for audio_file_name, task in task_tracking.items():
                if task["status"] == TranscriptionStatus.IN_PROGRESS:
                    print(
                        "[bold red]The transcription of the audio file"
                        f" {audio_file_name} failed.[/bold red]"
                    )

            split_progress.advance(split_progress_task_id)

        # Returns a counter of the number of files that were successfully transcribed and
        # the number of files that failed to be transcribed.
        status_counts = Counter(task["status"] for task in task_tracking.values())

        return ProviderResult(
            provider_name=self.__class__.__name__,
            completed=status_counts[TranscriptionStatus.COMPLETED],
            failed=status_counts[TranscriptionStatus.FAILED],
        )

    async def _save_result(
        self, audio_file_name: str, asr_output: dict, output_dir: Path
    ) -> None:
        """Save the result to a file."""
        _file_name = audio_file_name.split(".")[0]
        file_path = (
            output_dir / f"{self.__class__.__name__.lower()}" / f"{_file_name}.txt"
        )
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, mode="w") as f:
            await f.write(json.dumps(asr_output, indent=4, ensure_ascii=False))

    @abstractmethod
    async def _launch(
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
        self.options = AssemblyAIOptions(**options)

    async def _launch(self) -> None:
        """Call the API of the AssemblyAI ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Aws(ASRProvider):
    """The ASR provider class for AWS."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = AwsOptions(**options)

    async def _launch(self) -> None:
        """Call the API of the AWS ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Azure(ASRProvider):
    """The ASR provider class for Azure."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = AzureOptions(**options)

    async def _launch(self) -> None:
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

    async def _launch(
        self, audio_file: Path, url: str, headers: dict, session: aiohttp.ClientSession
    ) -> dict:
        """Run the Deepgram ASR provider."""
        async with aiofiles.open(audio_file, mode="rb") as f:
            async with session.post(url=url, data=f, headers=headers) as response:
                content = (await response.text()).strip()

        if not content:
            _status = TranscriptionStatus.FAILED
            body = None
        else:
            body = json.loads(content)

            if body.get("error"):
                _status = TranscriptionStatus.FAILED
            else:
                _status = TranscriptionStatus.COMPLETED

        return audio_file.name, _status, body

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Google(ASRProvider):
    """The ASR provider class for Google."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = GoogleOptions(**options)

    async def _launch(
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
        self.options = RevAIOptions(**options)

    async def _launch(self) -> None:
        """Call the API of the RevAI ASR provider."""
        pass

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass


class Speechmatics(ASRProvider):
    """The ASR provider class for Speechmatics."""

    def __init__(self, api_url: str, api_key: str, options: dict) -> None:
        super().__init__(api_url, api_key)
        self.options = SpeechmaticsOptions(**options)

    async def _launch(self) -> None:
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
        self.options = WordcabOptions(**options)

    async def _launch(
        self, audio_file: Path, url: str, headers: dict, session: aiohttp.ClientSession
    ) -> None:
        """Run the Wordcab ASR provider."""
        async with aiofiles.open(audio_file, mode="rb") as f:
            async with session.post(url=url, data=f, headers=headers) as response:
                content = (await response.text()).strip()

        if not content:
            _status = TranscriptionStatus.FAILED
            body = None
        else:
            body = json.loads(content)

            if body.get("detail"):
                _status = TranscriptionStatus.FAILED
            else:
                _status = TranscriptionStatus.COMPLETED

        return audio_file.name, _status, body

    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        pass
