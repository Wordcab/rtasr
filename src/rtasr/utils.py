"""Utils functions for rtasr."""

import asyncio
import json
import re
import string
import urllib.parse
import wave
import zipfile
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union

import aiofiles
import aiohttp
import dotenv
import numpy as np
from aiopath import AsyncPath
from pydantic import BaseModel
from rich import print
from rich.console import Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class ProviderPricing(BaseModel):
    """Provider pricing."""

    value: float
    unit: str


def attach_punctuation_to_last_word(sentence: str) -> str:
    """
    Attach punctuation to the last word of a sentence.

    Args:
        sentence (str):
            Sentence to attach punctuation to.

    Returns:
        Sentence with punctuation attached to the last word.
    """
    if isinstance(sentence, str) and sentence != "":
        words = sentence.split()

        sentence = ""
        for word in words:
            if word in string.punctuation:
                sentence += word
            else:
                sentence += " " + word

    return sentence.strip()


def build_query_string(params: Mapping[str, Any] = None) -> str:
    """
    Build a query string from a dictionary of parameters for API calls.

    Args:
        params (Mapping[str, Any]):
            Dictionary of parameters for API calls.

    Returns:
        Query string.
    """
    if params is None:
        params = {}

    filtered_parameters: List[Tuple[str, str]] = []
    for key, value in params.items():
        if value is None or value == "":
            continue
        else:
            filtered_parameters.append((key, str(value).lower()))

    return (
        "?" + urllib.parse.urlencode(filtered_parameters) if filtered_parameters else ""
    )


def create_live_panel(
    bar_width: int = 20,
    finished_text: str = "✅",
    spinner: str = "dots",
    spinner_speed: float = 0.5,
) -> Tuple[Progress, Progress, Progress, Group]:
    """
    Create a live panel for the progress bar.

    Args:
        bar_width (int):
            Width of the progress bar. Defaults to 20.
        finished_text (str):
            Text to display when the progress bar is finished.
            Defaults to "✅".
        spinner (str):
            The spinner type. Defaults to "dots".
        spinner_speed (float):
            The speed of the spinner. Defaults to 0.5.

    Returns:
        Tuple[Progress, Progress, Progress, Group]:
            The current progress, step progress, splits progress,
            and the progress group.
    """
    current_progress = Progress(TimeElapsedColumn(), TextColumn("{task.description}"))
    step_progress = Progress(
        TextColumn("  "),
        TimeElapsedColumn(),
        SpinnerColumn(spinner, finished_text=finished_text, speed=spinner_speed),
        TextColumn("[bold purple]{task.fields[action]}"),
        BarColumn(bar_width=bar_width),
    )
    splits_progress = Progress(
        TextColumn("[bold blue]Progres: {task.percentage:.0f}%"),
        BarColumn(),
        TextColumn("({task.completed} of {task.total} steps done)"),
    )

    progress_group = Group(
        Panel(Group(current_progress, step_progress, splits_progress))
    )

    return current_progress, step_progress, splits_progress, progress_group


async def download_file(
    url: str,
    output_dir: Path,
    session: aiohttp.ClientSession,
    use_cache: bool,
    target_name: Union[str, None] = None,
) -> Path:
    """
    Download a file from url to output_dir.

    Args:
        url (str):
            URL to download from.
        output_dir (Path):
            Path to output directory.
        session (aiohttp.ClientSession):
            aiohttp session to use for downloading.`
        use_cache (bool):
            Whether to use the cache or not.
        target_name (Union[str, None]):
            Target name of the file. Defaults to None.

    Returns:
        Path to output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_name is not None:
        _output_file = output_dir / target_name
    else:
        _output_file = output_dir / Path(url).name

    output_file = _filename_dots_filter(_output_file)
    if use_cache and output_file.exists():
        return output_file
    else:
        output_file.unlink(missing_ok=True)

    async with session.get(url) as response:
        with output_file.open("wb") as f:
            while True:
                chunk = await response.content.read(1024)

                if not chunk:
                    break

                f.write(chunk)

    return output_file


def get_api_key(provider: str) -> Union[str, None]:
    """Get the API key for the provider.

    All the API keys must be stored in a .env file in the root directory of the
    project.

    Args:
        provider (str):
            The provider to get the API key for.

    Returns:
        The API key for the provider or None if not found.
    """
    config = dotenv.dotenv_values(".env")

    key = config.get(f"{provider.upper()}_API_KEY", None)
    if (
        key is None
        or key == ""
        or key == "<your key here>"
        and provider != "wordcab-hosted"
    ):
        print(
            f"No API key found for {provider.upper()}. "
            f"Please add `{provider.upper()}_API_KEY` to the `.env` file."
        )
        return None

    return key


def get_audio_duration_from_samples(
    samples: Union[int, np.ndarray], sample_rate: int
) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        samples (Union[int, np.ndarray]):
            Samples of audio file.
        sample_rate (int):
            Sample rate of audio file.

    Returns:
        Duration of audio file in seconds.
    """
    if isinstance(samples, np.ndarray):
        _samples = len(samples)
    elif isinstance(samples, int):
        _samples = samples

    return _samples / sample_rate


def get_audio_duration_from_file(filename: Path) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        filename (Path):
            Path to audio file.

    Returns:
        Duration of audio file in seconds.
    """
    with wave.open(str(filename), "rb") as audio_file:
        num_frames = audio_file.getnframes()
        frame_rate = audio_file.getframerate()

    return num_frames / frame_rate


def get_human_readable_duration(seconds: Union[float, List[float]]) -> str:
    """
    Convert seconds to human readable duration.

    Args:
        seconds (Union[float, List[float]]):
            Duration in seconds or list of durations in seconds.

    Returns:
        Human readable duration.
    """
    if isinstance(seconds, list):
        seconds = np.sum(seconds)

    m, s = divmod(seconds, 60)

    return f"{int(m):02d}:{int(s):02d}"


def get_human_readable_price(seconds: Union[float, List[float]], pricing: dict) -> str:
    """
    Convert seconds to human readable price.

    Args:
        seconds (Union[float, List[float]]):
            Duration in seconds or list of durations in seconds.
        pricing (dict):
            Pricing of provider containing value and unit.

    Returns:
        Human readable price.
    """
    if isinstance(seconds, list):
        seconds = np.sum(seconds)

    _pricing = ProviderPricing(**pricing)

    if _pricing.unit == "minute":
        price = _pricing.value * seconds / 60
    elif _pricing.unit == "hour":
        price = _pricing.value * seconds / 3600
    else:
        price = _pricing.value * seconds

    return f"${price:.2f}"


def get_files(path: Path) -> Path:
    """
    Get all files in a directory.

    Args:
        path (Path):
            Path to directory to get files from.

    Returns:
        Generator of files in directory.
    """
    for p in path.iterdir():
        if p.is_file():
            yield p


async def unzip_file(zip_path: Path, output_dir: Path, use_cache: bool = True) -> Path:
    """
    Unzip a file to a directory.

    Args:
        zip_path (Path):
            Path to zip file.
        output_dir (Path):
            Path to output directory.
        use_cache (bool):
            Whether to use the cache or not. Defaults to True.

    Returns:
        Path to output directory.
    """

    def extract_zip_sync(zip_path: Path, extract_to: Path) -> None:
        """Synchronous function to extract ZIP."""
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_to)

    output_dir.mkdir(parents=True, exist_ok=True)
    unzip_path = output_dir / zip_path.stem

    if use_cache and unzip_path.exists():
        pass
    else:
        unzip_path.unlink(missing_ok=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract_zip_sync, zip_path, output_dir)

    return unzip_path


def reconstruct_acronym(text: str) -> str:
    """
    Reconstruct an acronym by removing the underscores

    Args:
        text (str):
            Text to reconstruct.

    Returns:
        Reconstructed text.
    """
    pattern = r"(?<=[A-Z])_(?=[A-Z]|[a-z]| )"
    result = re.sub(pattern, "", text)

    return result


def remove_bracketed_text(sentence: str) -> str:
    """
    Remove bracketed text from a sentence.

    Args:
        sentence (str):
            Sentence to remove bracketed text from.

    Returns:
        Sentence with bracketed text removed.
    """
    return re.sub(r"<[^>]+>", "", sentence)


def resolve_cache_dir() -> Path:
    """Resolve the cache directory for rtasr."""
    return Path.home() / ".cache" / "rtasr"


async def store_evaluation_results(
    results: dict,
    save_path: AsyncPath,
) -> None:
    """Store the evaluation results in a JSON file."""
    results.pop("status")

    await save_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(save_path, mode="w") as file:
        await file.write(json.dumps(results, indent=4, ensure_ascii=False))


def _ami_speaker_list(ami_rttm_segments: List[List[Union[str, float]]]) -> List[str]:
    """
    Get the list of speakers from the AMI RTTM segments and keep the order
    of appearance.

    Args:
        ami_rttm_segments (List[Tuple[str, float, float]]):
            List of RTTM segments from the AMI dataset.

    Returns:
        List of speakers.
    """
    speaker_list: List[str] = []

    for segment in ami_rttm_segments:
        speaker = segment[0]

        if speaker not in speaker_list:
            speaker_list.append(speaker)

    return speaker_list


async def _check_cache(
    file_name: str,
    evaluation_dir: AsyncPath,
    split: str,
    provider: str,
    metric: str,
) -> Tuple[bool, AsyncPath]:
    """Check the cache for the results of the diarization evaluation.

    This method check if the provider has already been evaluated for the given
    audio file. If so, it returns True, otherwise it returns False.

    Args:
        file_name (str):
            The file name to check.
        evaluation_dir (Path):
            The evaluation directory where the results are saved, i.e. the cache.
        split (str):
            The split of the dataset.
        provider (str):
            The provider to check.
        metric (str):
            The metric to check.

    Returns:
        Tuple[bool, Path]:
            A tuple containing a boolean indicating whether the provider has
            already been evaluated for the given audio file and the path to the
            evaluation results.
    """
    _file_name = file_name.split(".")[0]
    eval_output_file_path = AsyncPath(
        evaluation_dir / split / provider / metric / f"{_file_name}.json"
    )

    eval_output_exists = await eval_output_file_path.exists()

    return eval_output_exists, eval_output_file_path


def _filename_dots_filter(file_path: Path) -> Path:
    """
    Filter dots in the filename to avoid issues with API calls.

    Args:
        file_path (Path):
            Path to file.

    Returns:
        Path to file with dots replaced with underscores.
    """
    filename = file_path.name
    new_filename = filename.replace(".", "_", filename.count(".") - 1)
    new_file_path = file_path.with_name(new_filename)

    return new_file_path
