"""Diarization Error Rate (DER) evaluation implementation."""

import asyncio
import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import aiofiles
import spyder
from aiopath import AsyncPath
from pydantic import BaseModel
from rich import print
from rich.progress import Progress, TaskID
from typing_extensions import Literal

from rtasr.constants import DATASETS, PROVIDERS
from rtasr.speaker_map import AMISpeakerMap
from rtasr.utils import _ami_speaker_list


class DerEvalMode(tuple, Enum):
    """The DER evaluation mode.

    Note:
        The evaluation mode is a tuple of (collar, ignore_overlap). There are
        three evaluation modes available:
        * FULL:
            the DIHARD challenge style evaluation, the most strict way of
            evaluating diarization (collar, ignore_overlap) = (0.0, False).
        * FAIR:
            the evaluation setup used in VoxSRC challenge, more permissive
            than the previous one (collar, ignore_overlap) = (0.25, False).
        * FORGIVING:
            the traditional evaluation setup, more permissive than the two
            previous ones (collar, ignore_overlap) = (0.25, True).

    Attributes:
        collar (float):
            The collar value to use.
        ignore_overlap (bool):
            Whether to ignore overlapped speech or not (i.e. speech segments
            that are not annotated as overlapped speech but that overlap with
            other speech segments).
    """

    FULL = (0.0, False)
    FAIR = (0.25, False)
    FORGIVING = (0.25, True)


class ProviderDerResult(BaseModel):
    """The DER evaluation result for a provider."""

    cached: int
    evaluated: int
    not_found: int
    provider_name: str


class DerResult(BaseModel):
    """The DER evaluation result."""

    split_name: str
    results: List[ProviderDerResult]


class EvaluationStatus(str, Enum):
    """Status of the evaluation."""

    CACHED = "CACHED"
    EVALUATED = "EVALUATED"
    IN_PROGRESS = "IN_PROGRESS"
    NOT_FOUND = "NOT_FOUND"


class DerTaskStatus(str, Enum):
    """Status of a DER evaluation task."""

    DONE = "DONE"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"


async def evaluate_der(
    dataset: str,
    split_name: str,
    split_rttm_files: List[Path],
    evaluation_dir: Path,
    transcription_dir: Path,
    split_progress: Progress,
    split_progress_task_id: TaskID,
    step_progress: Progress,
    use_cache: bool,
    debug: bool,
) -> DerResult:
    """
    Evaluate the Diarization Error Rate (DER).
    TODO: Add docstrings
    TODO: Implement cache
    """
    if debug:
        split_rttm_files = split_rttm_files[:1]

    step_progress_task_id = step_progress.add_task(
        "",
        action=f"[bold green]{split_name}[/bold green]",
        total=len(split_rttm_files),
    )

    providers = list(PROVIDERS.keys())
    task_tracking: Dict[str, Any] = {}
    tasks: List[Callable] = []

    for rttm_file in split_rttm_files:
        task_tracking[rttm_file.name] = {
            "status": DerTaskStatus.IN_PROGRESS,
            "rttm_file_name": rttm_file.name,
            "split": split_name,
        }
        tasks.append(
            compute_score(
                dataset=dataset,
                ref_rttm_path=rttm_file,
                providers=providers,
                split=split_name,
                transcription_dir=transcription_dir,
                step_progress=step_progress,
            )
        )

    for future in asyncio.as_completed(tasks):
        try:
            task_result = await future
        except Exception as e:
            raise Exception(e) from e
        finally:
            step_progress.advance(step_progress_task_id)

    step_progress.update(step_progress_task_id, advance=len(split_rttm_files))
    split_progress.advance(split_progress_task_id)

    # status_counts = Counter(task["status"] for task in task_tracking.values())

    return DerResult(
        split_name=split_name,
        results=[],
    )


async def compute_score(
    dataset: str,
    ref_rttm_path: AsyncPath,
    providers: List[str],
    split: str,
    transcription_dir: Path,
    step_progress: Progress,
) -> None:
    """Compute the score for each provider."""
    # Read the reference RTTM file and convert it to a segment
    # with proper speaker mapping
    ref_rttm_content: List[List[Union[str, float]]] = await _prepare_rttm_content(ref_rttm_path)

    ref_rttm: List[Tuple[str, float, float]] = await _prepare_provider_rttm_segments(
        rttm_content=ref_rttm_content,
        target_name=dataset,
        target_type="dataset",
    )

    # For each provider:
    #   Read the the rttm file if available and convert it to a segment
    #   Score the reference and hypothesis segments
    for provider in providers:
        provider_rttm_path = AsyncPath(
            transcription_dir / split / provider / "rttm" / ref_rttm_path.name
        )
        if await provider_rttm_path.exists():
            hyp_rttm_content = await _prepare_rttm_content(provider_rttm_path)
            hyp_rttm = await _prepare_provider_rttm_segments(
                rttm_content=hyp_rttm_content,
                target_name=provider,
                target_type="provider",
            )

            print(spyder.DER(ref_rttm, hyp_rttm, collar=0.0, regions="single"))

    # Gather the scores for each provider on one rttm file
    # Save the scores in a json file and csv file

async def _prepare_rttm_content(
    rttm_path: Union[str, Path, AsyncPath]
) -> List[List[Union[str, float]]]:
    """
    Prepare the RTTM content for evaluation. The RTTM content is a list of
    segments. We only keep the speaker, start and end time of each segment.

    Args:
        rttm_path (Union[str, Path, AsyncPath]):
            Path to the RTTM file.

    Returns:
        List of RTTM segments (speaker, start, end).
    """
    if isinstance(rttm_path, (str, Path)):
        rttm_path = AsyncPath(rttm_path)

    async with aiofiles.open(rttm_path, mode="r") as file:
        content: List[str] = (await file.read()).splitlines()

    rttm_content: List[List[Union[str, float]]] = []
    for line in content:
        items = line.split()
        rttm_content.append(
            [
                str(items[7]),
                float(items[3]),
                float(items[3]) + float(items[4]),
            ]
        )

    return rttm_content


async def _prepare_provider_rttm_segments(
    rttm_content: List[List[Union[str, float]]],
    target_name: str,
    target_type: Literal["dataset", "provider"],
) -> List[Tuple[str, float, float]]:
    """
    Prepare the RTTM segments for evaluation for datasets and providers.

    Args:
        rttm_content (List[List[Union[str, float]]]):
            List of RTTM segments (speaker, start, end).
        target_name (str):
            Name of the dataset or provider.
        target_type (Literal["dataset", "provider"]):
            Type of the target (dataset or provider).

    Returns:
        List[Tuple[str, float, float]]: 
            List of RTTM segments (speaker, start, end) with proper speaker
            mapping.
    """
    if target_type == "dataset" and target_name == "ami":
        speaker_list = _ami_speaker_list(rttm_content)
        speaker_map = AMISpeakerMap(speaker_list=speaker_list)
    else:
        _constants = DATASETS if target_type == "dataset" else PROVIDERS
        speaker_map = getattr(
            importlib.import_module("rtasr.speaker_map"),
            _constants[target_name]["speaker_map"],
        )

    prepared_ref_rttm: List[Tuple[str, float, float]] = [
        (speaker_map.from_value(item[0]), item[1], item[2])
        for item in rttm_content
    ]

    return prepared_ref_rttm
