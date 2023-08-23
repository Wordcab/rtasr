"""Diarization Error Rate (DER) evaluation implementation."""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List

from aiopath import AsyncPath
from pydantic import BaseModel
from rich import print
from rich.progress import Progress, TaskID


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


async def evaluate_der(
    split_name: str,
    split_rttm_files: List[Path],
    providers: List[str],
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
        pass

    step_progress_task_id = step_progress.add_task(
        "",
        action=f"[bold green]{split_name}[/bold green]",
        total=len(split_rttm_files),
    )

    task_tracking: Dict[str, Any] = {}
    tasks: List[Callable] = []

    for rttm_file in split_rttm_files:
        task_tracking[rttm_file.name] = {
            "status": EvaluationStatus.IN_PROGRESS,
            "rttm_file_name": rttm_file.name,
            "split": split_name,
        }
        tasks.append(
            compute_score(
                ref_rttm=rttm_file,
                providers=providers,
                split=split_name,
                transcription_dir=transcription_dir,
                step_progress=step_progress,
            )
        )

    for future in asyncio.as_completed(tasks):
        try:
            task_result = await future
            print(task_result)
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
    ref_rttm: AsyncPath,
    providers: List[str],
    split: str,
    transcription_dir: Path,
    step_progress: Progress,
) -> None:
    """Compute the score for each provider."""
    pass

    # Read the reference RTTM file and convert it to a segment

    # For each provider:
    #   Read the the rttm file if available and convert it to a segment
    #   Score the reference and hypothesis segments

    # Gather the scores for each provider on one rttm file
    # Save the scores in a json file and csv file
