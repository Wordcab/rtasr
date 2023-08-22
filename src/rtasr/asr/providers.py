"""Providers are the classes that actually do the API calls to the different ASR services."""

import asyncio
import json
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

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
from rtasr.asr.schemas import (
    ASROutput,
    AssemblyAIOutput,
    AwsOutput,
    AzureOutput,
    DeepgramOutput,
    GoogleOutput,
    RevAIOutput,
    SpeechmaticsOutput,
    WordcabOutput,
    WordcabTranscript,
)
from rtasr.concurrency import ConcurrencyHandler, ConcurrencyToken
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

    def __init__(
        self, api_url: str, api_key: str, concurrency_limit: Union[int, None]
    ) -> None:
        """
        Initialize the ASR provider.

        Args:
            api_url (str):
                The URL of the ASR provider API.
            api_key (str):
                The API key of the ASR provider.
            concurrency_limit (Union[int, None]):
                The maximum number of concurrent API calls. If `None`, there is no
                limit.
        """
        self.config = ProviderConfig(api_url=api_url, api_key=api_key)
        self.concurrency_handler = ConcurrencyHandler(limit=concurrency_limit)

    async def launch(
        self,
        audio_files: Dict[str, List[Path]],
        output_dir: Path,
        session: aiohttp.ClientSession,
        split_progress: Progress,
        split_progress_task_id: TaskID,
        step_progress: Progress,
    ) -> ProviderResult:
        """
        Call the API of the ASR provider.

        Args:
            audio_files (Dict[str, List[Path]]):
                The audio files to transcribe with the ASR provider. The keys are the
                names of the splits and the values are the list of audio files to
                transcribe for each split.
            output_dir (Path):
                The output directory where to save the results.
            session (aiohttp.ClientSession):
                The aiohttp session for the API calls.
            split_progress (Progress):
                The progress bar for the split progress. It is used to track the
                progress of the transcription of each split.
            split_progress_task_id (TaskID):
                The task ID of the split progress bar. It is used to update the
                progress of the split progress bar.
            step_progress (Progress):
                The progress bar for the step progress. It is used to track the
                progress of the transcription of all the audio files for a
                specific ASR provider.

        Returns:
            ProviderResult:
                The result of the ASR provider. It contains the name of the provider,
                the number of files that were successfully transcribed and the number
                of files that failed to be transcribed.
        """
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
                if test_counter < 1:  # TODO: Remove this line, test purpose only
                    tasks.append(
                        self._launch(
                            audio_file=audio_file,
                            url=self.config.api_url,
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
                    audio_file_name, status, asr_output = task_result

                    if status == TranscriptionStatus.COMPLETED:
                        task_tracking[audio_file_name]["status"] = status
                        _split = task_tracking[audio_file_name]["split"]

                        rttm_lines = await self.result_to_rttm(asr_output=asr_output)
                        await self._save_results(
                            audio_file_name=audio_file_name,
                            asr_output=asr_output,
                            rttm_lines=rttm_lines,
                            output_dir=output_dir / _split,
                        )

                    elif status == TranscriptionStatus.FAILED:
                        task_tracking[audio_file_name]["status"] = status
                        print(
                            rf"[bold red]\[{self.__class__.__name__}] ->"
                            rf" {asr_output}[/bold"
                            " red]"
                        )
                except Exception as e:
                    raise Exception(e) from e
                finally:
                    step_progress.advance(step_progress_task_id)

            split_progress.advance(split_progress_task_id)

        status_counts = Counter(task["status"] for task in task_tracking.values())

        return ProviderResult(
            provider_name=self.__class__.__name__,
            completed=status_counts[TranscriptionStatus.COMPLETED],
            failed=status_counts[TranscriptionStatus.FAILED],
        )

    async def _save_results(
        self,
        audio_file_name: str,
        asr_output: ASROutput,
        rttm_lines: List[str],
        output_dir: Path,
    ) -> None:
        """
        Save the asr outputs and the RTTM files to disk.

        Args:
            audio_file_name (str):
                The name of the audio file.
            asr_output (ASROutput):
                The output of the ASR provider to save.
            rttm_lines (List[str]):
                The lines of the RTTM file to save.
            output_dir (Path):
                The output directory where to save the results.
        """
        _file_name = audio_file_name.split(".")[0]
        asr_output_file_path = (
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "original"
            / f"{_file_name}.json"
        )
        if not asr_output_file_path.parent.exists():
            asr_output_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(asr_output_file_path, mode="w") as f:
            await f.write(
                json.dumps(asr_output.model_dump(), indent=4, ensure_ascii=False)
            )

        rttm_file_path = (
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "rttm"
            / f"{_file_name}.txt"
        )
        if not rttm_file_path.parent.exists():
            rttm_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(rttm_file_path, mode="w") as f:
            await f.write("\n".join(rttm_lines))

    @abstractmethod
    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        headers: dict,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, ASROutput]:
        """Run the ASR provider."""
        raise NotImplementedError(
            "The ASR provider must implement the `_launch` method."
        )

    @abstractmethod
    def result_to_rttm(self) -> None:
        """Convert the result to RTTM format."""
        raise NotImplementedError(
            "The ASR provider must implement the `result_to_rttm` method."
        )


class AssemblyAI(ASRProvider):
    """The ASR provider class for AssemblyAI."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = AssemblyAIOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, AssemblyAIOutput]:
        """Call the API of the AssemblyAI ASR provider."""
        headers = {
            "Authorization": f"{self.config.api_key.get_secret_value()}",
        }

        concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

        async with aiofiles.open(audio_file, mode="rb") as f:
            async with session.post(
                url=f"{url}/upload{build_query_string(self.options)}",
                data=f,
                headers=headers,
            ) as response:
                content = (await response.text()).strip()

        upload_url = json.loads(content).get("upload_url")
        payload = {"audio_url": upload_url}

        async with session.post(
            url=f"{url}/transcript", json=payload, headers=headers
        ) as response:
            content = (await response.text()).strip()

        transcript_id = json.loads(content).get("id")

        while True:
            async with session.get(
                url=f"{url}/transcript/{transcript_id}", headers=headers
            ) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            if body.get("status") == "completed":
                asr_output = AssemblyAIOutput.from_json(body)
                _status = TranscriptionStatus.COMPLETED
                break
            elif body.get("status") == "error":
                asr_output = body.get("error")
                _status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    def result_to_rttm(self, asr_output: AssemblyAIOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class Aws(ASRProvider):
    """The ASR provider class for AWS."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = AwsOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, AwsOutput]:
        """Call the API of the AWS ASR provider."""
        concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

        self.concurrency_handler.put(concurr_token)

        return None

    def result_to_rttm(self, asr_output: AwsOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class Azure(ASRProvider):
    """The ASR provider class for Azure."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = AzureOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, AzureOutput]:
        """Call the API of the Azure ASR provider."""
        pass

    def result_to_rttm(self, asr_output: AzureOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class Deepgram(ASRProvider):
    """The ASR provider class for Deepgram."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        """Initialize the Deepgram ASR provider."""
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = DeepgramOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, DeepgramOutput]:
        """Run the Deepgram ASR provider."""
        headers = {
            "Authorization": f"Token {self.config.api_key.get_secret_value()}",
            "Content-Type": f"audio/{audio_file.suffix[1:]}",
        }

        concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

        _url = f"{url}{build_query_string(self.options)}"
        async with aiofiles.open(audio_file, mode="rb") as f:
            async with session.post(url=_url, data=f, headers=headers) as response:
                content = (await response.text()).strip()

        if not content:
            _status = TranscriptionStatus.FAILED
            asr_output = None
        else:
            body = json.loads(content)

            if body.get("err_code"):
                asr_output = body.get("err_msg")
                _status = TranscriptionStatus.FAILED
            else:
                asr_output = DeepgramOutput.from_json(body)
                _status = TranscriptionStatus.COMPLETED

        self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    def result_to_rttm(self, asr_output: DeepgramOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class Google(ASRProvider):
    """The ASR provider class for Google."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = GoogleOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, GoogleOutput]:
        """Run the ASR provider."""
        pass

    def result_to_rttm(self, asr_output: GoogleOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class RevAI(ASRProvider):
    """The ASR provider class for RevAI."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = RevAIOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, RevAIOutput]:
        """Call the API of the RevAI ASR provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
        }

        concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("media", f, filename=audio_file.name)
            form.add_field("options", json.dumps(self.options, sort_keys=True))

            async with session.post(
                url=f"{url}/jobs/", data=form, headers=headers
            ) as response:
                content = (await response.text()).strip()

        body = json.loads(content)
        job_id = body.get("id")

        while True:
            async with session.get(
                url=f"{url}/jobs/{job_id}", headers=headers
            ) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            if body.get("status") == "transcribed":
                _status = TranscriptionStatus.COMPLETED
                break
            elif body.get("status") == "failed":
                _status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        if _status == TranscriptionStatus.COMPLETED:
            async with session.get(
                url=f"{url}/jobs/{job_id}/transcript", headers=headers
            ) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            asr_output = RevAIOutput.from_json(body)

        else:
            asr_output = body.get("failure_detail")

        self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    def result_to_rttm(self, asr_output: RevAIOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class Speechmatics(ASRProvider):
    """The ASR provider class for Speechmatics."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = SpeechmaticsOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, SpeechmaticsOutput]:
        """Call the API of the Speechmatics ASR provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
        }

        concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

        _url = f"{url}/jobs"
        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("data_file", f, filename=audio_file.name)
            form.add_field("config", json.dumps(self.options, ensure_ascii=False))

            async with session.post(url=_url, data=form, headers=headers) as response:
                content = (await response.text()).strip()

        body = json.loads(content)
        job_id = body.get("id")

        while True:
            async with session.get(url=f"{_url}/{job_id}", headers=headers) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            _job = body.get("job")

            if _job.get("status") == "done":
                _status = TranscriptionStatus.COMPLETED
                break
            elif _job.get("status") == "rejected":
                _status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        if _status == TranscriptionStatus.COMPLETED:
            async with session.get(
                url=f"{_url}/{job_id}/transcript?format=json-v2", headers=headers
            ) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            asr_output = SpeechmaticsOutput.from_json(body)

        else:
            _errors = body.get("errors")
            asr_output = "\n".join([error.get("message") for error in _errors])

        self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    def result_to_rttm(self, asr_output: SpeechmaticsOutput) -> None:
        """Convert the result to RTTM format."""
        pass


class Wordcab(ASRProvider):
    """The ASR provider class for Wordcab."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        options: dict,
        concurrency_limit: Union[int, None],
    ) -> None:
        """Initialize the Wordcab ASR provider."""
        super().__init__(api_url, api_key, concurrency_limit)
        self.options = WordcabOptions(**options)

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, WordcabOutput]:
        """Run the Wordcab ASR provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Disposition": f'attachment; filename="{audio_file.name}"',
        }

        concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

        _url = f"{url}/transcribe{build_query_string(self.options)}"
        async with aiofiles.open(audio_file, mode="rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=audio_file.name)

            async with session.post(url=_url, data=form, headers=headers) as response:
                content = (await response.text()).strip()

        body = json.loads(content)
        job_name = body.get("job_name")
        transcript_id = body.get("transcript_id")

        while True:
            async with session.get(
                url=f"{url}/jobs/{job_name}", headers=headers
            ) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            if body.get("job_status") == "TranscriptComplete":
                _status = TranscriptionStatus.COMPLETED
                break
            elif body.get("job_status") == "Error":
                _status = TranscriptionStatus.FAILED
                break
            else:
                await asyncio.sleep(3)

        if _status == TranscriptionStatus.COMPLETED:
            async with session.get(
                url=f"{url}/transcripts/{transcript_id}", headers=headers
            ) as response:
                content = (await response.text()).strip()

            body = json.loads(content)
            asr_output = WordcabOutput.from_json(body)

        else:
            asr_output = None

        self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: WordcabOutput) -> List[str]:
        """Convert the result to RTTM format."""
        all_transcripts: List[WordcabTranscript] = asr_output.transcript

        rttm_lines: List[str] = []
        for transcript in all_transcripts:
            start_seconds = transcript.timestamp_start / 1000
            end_seconds = transcript.timestamp_end / 1000
            speaker = transcript.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines
