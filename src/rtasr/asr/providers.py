"""Providers are the classes that actually do the API calls to the different ASR services."""

import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import aiofiles
import aiohttp
from aiopath import AsyncPath
from pydantic import BaseModel, HttpUrl, SecretStr
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
    AssemblyAIUtterance,
    AssemblyAIWord,
    AwsOutput,
    AzureOutput,
    DeepgramOutput,
    DeepgramUtterance,
    GoogleOutput,
    RevAIElement,
    RevAIMonologue,
    RevAIOutput,
    SpeechmaticsOutput,
    SpeechmaticsResult,
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

    cached: int
    completed: int
    errors: List[str]
    failed: int
    provider_name: str


class TranscriptionStatus(str, Enum):
    """Status of the transcription."""

    CACHED = "CACHED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"


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
        self.max_retries = 3

    @property
    @abstractmethod
    def output_schema(self) -> ASROutput:
        """The output schema of the ASR provider."""
        return ASROutput

    async def launch(
        self,
        audio_files: Dict[str, List[Path]],
        output_dir: Path,
        session: aiohttp.ClientSession,
        split_progress: Progress,
        split_progress_task_id: TaskID,
        step_progress: Progress,
        use_cache: bool,
        debug: bool,
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
            use_cache (bool):
                Whether to use the cache or not. If `True`, the ASR provider will
                not transcribe the audio files that are already in the cache.
                The RTTM files will be generated from the cache if not already
                present.
            debug (bool):
                Whether to run in debug mode or not. If `True`, the ASR provider
                will only transcribe the first audio file of each split.
                This is useful for debugging or testing the full process.

        Returns:
            ProviderResult:
                The result of the ASR provider. It contains the name of the provider,
                the number of files that were successfully transcribed and the number
                of files that failed to be transcribed.
        """
        if debug:
            audio_files = {
                split_name: audio_files[split_name][0:5]
                for split_name in audio_files.keys()
            }

        task_tracking: Dict[str, Any] = {}

        for split_name, split_audio_files in audio_files.items():
            tasks: List[Callable] = []
            for audio_file in split_audio_files:
                task_tracking[audio_file.name] = {
                    "audio_file_name": audio_file.name,
                    "error": None,
                    "split": split_name,
                    "status": TranscriptionStatus.IN_PROGRESS,
                }
                _check_cache_task = await self._check_cache(
                    audio_file=AsyncPath(audio_file),
                    output_dir=AsyncPath(output_dir / split_name),
                )
                asr_output_exists, rttm_file_exists = _check_cache_task

                if use_cache and asr_output_exists:
                    task_tracking[audio_file.name]["asr_output_cache"] = True

                    if not rttm_file_exists:
                        task_tracking[audio_file.name]["rttm_cache"] = False
                        tasks.append(
                            self._get_asr_output_from_cache(
                                audio_file=audio_file,
                                output_dir=output_dir / split_name,
                            )
                        )
                    else:
                        task_tracking[audio_file.name][
                            "status"
                        ] = TranscriptionStatus.CACHED
                        task_tracking[audio_file.name]["rttm_cache"] = True

                else:
                    task_tracking[audio_file.name]["asr_output_cache"] = False
                    task_tracking[audio_file.name]["rttm_cache"] = False
                    tasks.append(
                        self._launch(
                            audio_file=audio_file,
                            url=self.config.api_url,
                            session=session,
                        )
                    )

            step_progress_task_id = step_progress.add_task(
                "",
                action=f"[bold green]{self.__class__.__name__}[/bold green]",
                total=len(tasks),
            )

            try:
                for future in asyncio.as_completed(tasks):
                    task_result = await future
                    audio_file_name, status, asr_output = task_result

                    if (
                        status == TranscriptionStatus.CACHED
                        or status == TranscriptionStatus.COMPLETED
                    ):
                        task_tracking[audio_file_name]["status"] = status
                        _split = task_tracking[audio_file_name]["split"]

                        if not task_tracking[audio_file_name]["rttm_cache"]:
                            rttm_lines = await self.result_to_rttm(
                                asr_output=asr_output
                            )
                            await self._save_rttm_files(
                                audio_file_name=audio_file_name,
                                rttm_lines=rttm_lines,
                                output_dir=output_dir / _split,
                            )

                        if not task_tracking[audio_file_name]["asr_output_cache"]:
                            await self._save_asr_outputs(
                                audio_file_name=audio_file_name,
                                asr_output=asr_output,
                                output_dir=output_dir / _split,
                            )

                    elif status == TranscriptionStatus.FAILED:
                        task_tracking[audio_file_name]["status"] = status
                        if isinstance(asr_output, Exception):
                            task_tracking[audio_file_name]["error"] = str(asr_output)

                    step_progress.advance(step_progress_task_id)

            except Exception as e:
                print(
                    f"[bold red]Problem with {self.__class__.__name__} -> {e}[/bold"
                    " red]]"
                )
                # If there is an exception, we mark all the tasks still in progress
                # as failed.
                for task in task_tracking.values():
                    if task["status"] == TranscriptionStatus.IN_PROGRESS:
                        task_tracking[task["audio_file_name"]][
                            "status"
                        ] = TranscriptionStatus.FAILED

        step_progress.update(step_progress_task_id, advance=len(audio_files))
        split_progress.advance(split_progress_task_id)

        status_counts = Counter(task["status"] for task in task_tracking.values())

        return ProviderResult(
            provider_name=self.__class__.__name__,
            completed=status_counts[TranscriptionStatus.COMPLETED],
            failed=status_counts[TranscriptionStatus.FAILED],
            cached=status_counts[TranscriptionStatus.CACHED],
            errors=[
                f"{task['audio_file_name']} -> {task['error']}"
                for task in task_tracking.values()
                if task["error"]
            ],
        )

    async def _check_cache(
        self, audio_file: AsyncPath, output_dir: AsyncPath
    ) -> Tuple[bool, bool]:
        """Check the cache for the audio file.

        This method check if the audio file has already been transcribed and is
        in the cache. It will check the asr output file and the RTTM file.

        Args:
            audio_file (Path):
                The audio file to check.
            output_dir (Path):
                The output directory where the results are saved, i.e. the cache.

        Returns:
            bool:
                Whether the audio file is already in the cache or not.
        """
        _file_name = audio_file.name.split(".")[0]
        asr_output_file_path = (
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "original"
            / f"{_file_name}.json"
        )
        rttm_file_path = (
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "rttm"
            / f"{_file_name}.rttm"
        )

        asr_output_exists = await asr_output_file_path.exists()
        rttm_exists = await rttm_file_path.exists()

        return asr_output_exists, rttm_exists

    async def _get_asr_output_from_cache(
        self, audio_file: Path, output_dir: Path
    ) -> Tuple[str, TranscriptionStatus, ASROutput]:
        """Get the asr output from the cache and return it.

        This function return the same output as the `_launch` method but it
        loads the asr output from the cache instead of calling the API.

        Args:
            audio_file (Path):
                The audio file to load.
            output_dir (Path):
                The output directory where the results are saved, i.e. the cache.

        Returns:
            Tuple[str, TranscriptionStatus, ASROutput]:
                The same output as the `_launch` method but with the asr output
                loaded from the cache.
        """
        raw_asr_output = await self._load_cache(audio_file, output_dir)

        asr_output = self.output_schema.from_json(raw_asr_output)

        return audio_file.name, TranscriptionStatus.CACHED, asr_output

    async def _load_cache(self, audio_file: Path, output_dir: Path) -> dict:
        """Load the cache for the audio file.

        This method load the asr output file from the cache.

        Args:
            audio_file (Path):
                The audio file to load.
            output_dir (Path):
                The output directory where the results are saved, i.e. the cache.

        Returns:
            dict:
                The raw asr output loaded from the cache.
        """
        _file_name = audio_file.name.split(".")[0]
        asr_output_file_path = (
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "original"
            / f"{_file_name}.json"
        )

        async with aiofiles.open(asr_output_file_path, mode="r") as f:
            data = await f.read()

        raw_data = json.loads(data)

        return raw_data

    async def _save_asr_outputs(
        self,
        audio_file_name: str,
        asr_output: ASROutput,
        output_dir: Path,
    ) -> None:
        """
        Save the asr outputs to disk.

        Args:
            audio_file_name (str):
                The name of the audio file.
            asr_output (ASROutput):
                The output of the ASR provider to save.
            output_dir (Path):
                The output directory where to save the results.
        """
        _file_name = audio_file_name.split(".")[0]
        asr_output_file_path = AsyncPath(
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "original"
            / f"{_file_name}.json"
        )
        if not await asr_output_file_path.parent.exists():
            await asr_output_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(asr_output_file_path, mode="w") as f:
            await f.write(
                json.dumps(asr_output.model_dump(), indent=4, ensure_ascii=False)
            )

    async def _save_rttm_files(
        self,
        audio_file_name: str,
        rttm_lines: List[str],
        output_dir: Path,
    ) -> None:
        """
        Save the RTTM files to disk.

        Args:
            audio_file_name (str):
                The name of the audio file.
            rttm_lines (List[str]):
                The RTTM lines to save.
            output_dir (Path):
                The output directory where to save the results.
        """
        _file_name = audio_file_name.split(".")[0]
        rttm_file_path = AsyncPath(
            output_dir
            / f"{self.__class__.__name__.lower()}"
            / "rttm"
            / f"{_file_name}.rttm"
        )
        if not await rttm_file_path.parent.exists():
            await rttm_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(rttm_file_path, mode="w") as f:
            await f.write("\n".join(rttm_lines))

    @abstractmethod
    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, ASROutput]:
        """Run the ASR provider."""
        raise NotImplementedError(
            "The ASR provider must implement the `_launch` method."
        )

    @abstractmethod
    async def result_to_rttm(self, asr_output: ASROutput) -> List[str]:
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

    @property
    def output_schema(self) -> AssemblyAIOutput:
        """The output format of the AssemblyAI ASR provider."""
        return AssemblyAIOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, Union[AssemblyAIOutput, Exception]]:
        """Call the API of the AssemblyAI ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                headers = {
                    "Authorization": f"{self.config.api_key.get_secret_value()}",
                }

                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

                async with aiofiles.open(audio_file, mode="rb") as f:
                    async with session.post(
                        url=f"{url}/upload",
                        data=f,
                        headers=headers,
                    ) as response:
                        content = (await response.text()).strip()

                upload_url = json.loads(content).get("upload_url")
                payload = {"audio_url": upload_url, **self.options}

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
                        asr_output = Exception(body.get("error"))
                        _status = TranscriptionStatus.FAILED
                        break
                    else:
                        await asyncio.sleep(3)

                retries = self.max_retries  # To break the while loop

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                break

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: AssemblyAIOutput) -> List[str]:
        """Convert the result to RTTM format."""
        utterances: List[AssemblyAIUtterance] = asr_output.utterances

        rttm_lines: List[str] = []
        if not utterances:  # This means there is only one speaker
            words: List[AssemblyAIWord] = asr_output.words
            rttm_lines.append(f"{words[0].start / 1000} {words[-1].end / 1000} A")
        else:
            for utterance in utterances:
                start_seconds: float = utterance.start / 1000
                end_seconds: float = utterance.end / 1000
                speaker: str = utterance.speaker

                rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


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

    @property
    def output_schema(self) -> AwsOutput:
        """The output format of the AWS ASR provider."""
        return AwsOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, AwsOutput]:
        """Call the API of the AWS ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()
                raise NotImplementedError("Aws not implemented.")

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: AwsOutput) -> List[str]:
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

    @property
    def output_schema(self) -> AzureOutput:
        """The output format of the Azure ASR provider."""
        return AzureOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, AzureOutput]:
        """Call the API of the Azure ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()
                raise NotImplementedError("Azure not implemented.")

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: AzureOutput) -> List[str]:
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

    @property
    def output_schema(self) -> DeepgramOutput:
        """The output format of the Deepgram ASR provider."""
        return DeepgramOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, DeepgramOutput]:
        """Run the Deepgram ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                headers = {
                    "Authorization": f"Token {self.config.api_key.get_secret_value()}",
                    "Content-Type": f"audio/{audio_file.suffix[1:]}",
                }

                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

                _url = f"{url}{build_query_string(self.options)}"
                async with aiofiles.open(audio_file, mode="rb") as f:
                    async with session.post(
                        url=_url, data=f, headers=headers
                    ) as response:
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

                retries = self.max_retries  # To break the while loop

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: DeepgramOutput) -> List[str]:
        """Convert the result to RTTM format."""
        utterances: List[DeepgramUtterance] = asr_output.results.utterances

        rttm_lines: List[str] = []
        for utterance in utterances:
            start_seconds: float = utterance.start
            end_seconds: float = utterance.end
            speaker: int = utterance.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


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

    @property
    def output_schema(self) -> GoogleOutput:
        """The output format of the Google ASR provider."""
        return GoogleOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, GoogleOutput]:
        """Run the ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()
                raise NotImplementedError("Google not implemented.")

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: GoogleOutput) -> List[str]:
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

    @property
    def output_schema(self) -> RevAIOutput:
        """The output format of the RevAI ASR provider."""
        return RevAIOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, RevAIOutput]:
        """Call the API of the RevAI ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
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

                retries = self.max_retries  # To break the while loop

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: RevAIOutput) -> List[str]:
        """Convert the result to RTTM format."""
        monologues: List[RevAIMonologue] = asr_output.monologues

        rttm_lines: List[str] = []
        for monologue in monologues:
            start_seconds, end_seconds = self._get_timestamps(
                elements=monologue.elements
            )
            speaker: int = monologue.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines

    @staticmethod
    def _get_timestamps(elements: List[RevAIElement]) -> Tuple[float, float]:
        """Retrieve the start and end timestamps of a monologue."""
        start_ts = 0
        end_ts = 0

        for i in range(len(elements)):
            if elements[i].type == "text":
                start_ts = elements[i].ts
                break

        for i in range(len(elements) - 1, -1, -1):
            if elements[i].type == "text":
                end_ts = elements[i].end_ts
                break

        return start_ts, end_ts


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

    @property
    def output_schema(self) -> SpeechmaticsOutput:
        """The output format of the Speechmatics ASR provider."""
        return SpeechmaticsOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, SpeechmaticsOutput]:
        """Call the API of the Speechmatics ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                }

                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

                _url = f"{url}/jobs"
                async with aiofiles.open(audio_file, mode="rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("data_file", f, filename=audio_file.name)
                    form.add_field(
                        "config", json.dumps(self.options, ensure_ascii=False)
                    )

                    async with session.post(
                        url=_url, data=form, headers=headers
                    ) as response:
                        content = (await response.text()).strip()

                body = json.loads(content)
                job_id = body.get("id")

                await asyncio.sleep(1)  # Extra pause for Speechmatics

                while True:
                    async with session.get(
                        url=f"{_url}/{job_id}", headers=headers
                    ) as response:
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
                        url=f"{_url}/{job_id}/transcript?format=json-v2",
                        headers=headers,
                    ) as response:
                        content = (await response.text()).strip()

                    body = json.loads(content)
                    asr_output = SpeechmaticsOutput.from_json(body)

                else:
                    _errors = body.get("errors")
                    asr_output = "\n".join([error.get("message") for error in _errors])

                retries = self.max_retries  # To break the while loop

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: SpeechmaticsOutput) -> List[str]:
        """Convert the result to RTTM format."""
        results: List[SpeechmaticsResult] = asr_output.results

        rttm_lines: List[str] = []
        for result in results:
            if result.type == "word":
                start_seconds: float = result.start_time
                end_seconds: float = result.end_time
                speaker: str = result.alternatives[0].speaker

                if speaker != "UU":
                    rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines


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

    @property
    def output_schema(self) -> WordcabOutput:
        """The output format of the Wordcab ASR provider."""
        return WordcabOutput

    async def _launch(
        self,
        audio_file: Path,
        url: HttpUrl,
        session: aiohttp.ClientSession,
    ) -> Tuple[str, TranscriptionStatus, WordcabOutput]:
        """Run the Wordcab ASR provider."""
        retries = 0

        while retries < self.max_retries:
            try:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                    "Content-Disposition": f'attachment; filename="{audio_file.name}"',
                }

                concurr_token: ConcurrencyToken = await self.concurrency_handler.get()

                _url = f"{url}/transcribe{build_query_string(self.options)}"
                async with aiofiles.open(audio_file, mode="rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("file", f, filename=audio_file.name)

                    async with session.post(
                        url=_url, data=form, headers=headers
                    ) as response:
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
                    asr_output = body.get("error_message")

                retries = self.max_retries  # To break the while loop

            except (
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ServerDisconnectedError,
            ) as e:
                if retries >= self.max_retries:
                    _status = TranscriptionStatus.FAILED
                    asr_output = Exception(f"{e}\n{traceback.format_exc()}")
                    break
                else:
                    retries += 1
                    print(
                        f"Retrying {audio_file.name} for {self.__class__.__name__}..."
                    )
                    await asyncio.sleep(3)

            except Exception as e:
                _status = TranscriptionStatus.FAILED
                asr_output = Exception(f"{e}\n{traceback.format_exc()}")

            finally:
                self.concurrency_handler.put(concurr_token)

        return audio_file.name, _status, asr_output

    async def result_to_rttm(self, asr_output: WordcabOutput) -> List[str]:
        """Convert the result to RTTM format."""
        all_transcripts: List[WordcabTranscript] = asr_output.transcript

        rttm_lines: List[str] = []
        for transcript in all_transcripts:
            start_seconds: float = transcript.timestamp_start / 1000
            end_seconds: float = transcript.timestamp_end / 1000
            speaker: str = transcript.speaker

            rttm_lines.append(f"{start_seconds} {end_seconds} {speaker}")

        return rttm_lines
