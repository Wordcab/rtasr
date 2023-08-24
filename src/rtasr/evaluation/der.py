"""Diarization Error Rate (DER) evaluation implementation."""

import asyncio
import importlib
import json
import traceback
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import aiofiles
import spyder
from aiopath import AsyncPath
from pydantic import BaseModel
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


class EvaluationStatus(str, Enum):
    """Status of the evaluation."""

    CACHED = "CACHED"
    EVALUATED = "EVALUATED"
    NOT_FOUND = "NOT_FOUND"


class DerTaskStatus(str, Enum):
    """Status of a DER evaluation task."""

    DONE = "DONE"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"


class ProviderDerResult(BaseModel):
    """The DER evaluation result for a provider."""

    cached: int
    evaluated: int
    not_found: int
    provider_name: str


class ProviderComputeScore(BaseModel):
    """The score computed for a provider."""

    confusion: Union[float, None]
    der: Union[float, None]
    false_alarm: Union[float, None]
    miss: Union[float, None]
    status: EvaluationStatus


class ComputeScores(BaseModel):
    """The result of a compute score task."""

    error: Union[str, None]
    filename: str
    scores: Union[Dict[str, ProviderComputeScore], None]


class DerResult(BaseModel):
    """The DER evaluation result."""

    errors: List[str]
    split_name: str
    results: List[ProviderDerResult]


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
        from pathlib import PosixPath

        _pa = PosixPath(
            "/Users/chainyo/.cache/rtasr/datasets/voxconverse/rttm/voxconverse-master/test/aepyx.rttm"
        )
        split_rttm_files = [_pa]

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
            "rttm_file_name": rttm_file.name,
            "status": DerTaskStatus.IN_PROGRESS,
            "provider_results": {provider: None for provider in providers},
        }
        tasks.append(
            compute_score(
                dataset=dataset,
                ref_rttm_path=rttm_file,
                providers=providers,
                split=split_name,
                transcription_dir=transcription_dir,
                step_progress=step_progress,
                use_cache=use_cache,
            )
        )

    for future in asyncio.as_completed(tasks):
        task_result: ComputeScores = await future

        filename = task_result.filename
        if task_result.error:
            task_tracking[filename]["status"] = DerTaskStatus.ERROR
            task_tracking[filename]["error"] = f"{filename} -> {task_result.error}"

        else:
            task_tracking[filename]["status"] = DerTaskStatus.DONE

            for provider in task_result.scores:
                status = task_result.scores[provider].status

                if (
                    status == EvaluationStatus.CACHED
                    or status == EvaluationStatus.EVALUATED
                ):
                    file_exists, file_path = await _check_cache(
                        rttm_file=filename,
                        evaluation_dir=evaluation_dir,
                        split=split_name,
                        provider=provider,
                    )

                    if use_cache and file_exists:
                        task_tracking[filename]["provider_results"][
                            provider
                        ] = EvaluationStatus.CACHED
                    else:
                        task_tracking[filename]["provider_results"][
                            provider
                        ] = EvaluationStatus.EVALUATED
                        await _store_evaluation_results(
                            results=task_result.scores[provider],
                            save_path=file_path,
                        )

                else:
                    task_tracking[filename]["provider_results"][
                        provider
                    ] = EvaluationStatus.NOT_FOUND

        step_progress.advance(step_progress_task_id)

    step_progress.update(step_progress_task_id, advance=len(split_rttm_files))
    split_progress.advance(split_progress_task_id)

    results: List[ProviderDerResult] = []
    for provider in providers:
        counter = Counter(
            [
                task_tracking[rttm_file.name]["provider_results"][provider]
                for rttm_file in split_rttm_files
            ]
        )
        results.append(
            ProviderDerResult(
                cached=counter[EvaluationStatus.CACHED],
                evaluated=counter[EvaluationStatus.EVALUATED],
                not_found=counter[EvaluationStatus.NOT_FOUND],
                provider_name=provider,
            )
        )

    errors: List[str] = [
        task_tracking[rttm_file.name]["error"]
        for rttm_file in split_rttm_files
        if task_tracking[rttm_file.name]["status"] == DerTaskStatus.ERROR
    ]

    return DerResult(
        errors=errors,
        split_name=split_name,
        results=results,
    )


async def compute_score(
    dataset: str,
    ref_rttm_path: AsyncPath,
    providers: List[str],
    split: str,
    transcription_dir: Path,
    step_progress: Progress,
    use_cache: bool,
) -> ComputeScores:
    """Compute the score for each provider.

    The score is computed using the Diarization Error Rate (DER) metric.
    Here is the process:
        Read reference RTTM file.
                |
        Format reference RTTM file with
        proper speaker mapping (e.g.: 1, 2, 3...).
                |
        For each provider:
            If provider RTTM file exists:
                Read provider RTTM file.
                            |
                Format provider RTTM file with
                proper speaker mapping (e.g.: 1, 2, 3...).
                            |
                Compute DER score.
                |
        Return scores.

    Args:
        dataset (str):
            Name of the dataset.
        ref_rttm_path (AsyncPath):
            Path to the reference RTTM file.
        providers (List[str]):
            List of providers to evaluate.
        split (str):
            Name of the split of the dataset (e.g.: "train", "dev", "test").
        transcription_dir (Path):
            Path to the directory containing the transcriptions.
        step_progress (Progress):
            Progress bar for the current step.
        use_cache (bool):
            Wether to use cache or not.

    Returns:
        ComputeScores:
            The result of the compute score task. It contains the filename
            (i.e. the name of the reference RTTM file) and the scores for each
            provider in the form of a dictionary:
            {
                "provider_1": ProviderComputerScore,
                "provider_2": ProviderComputerScore,
                ...
            }
    """
    try:
        step_progress_task_id = step_progress.add_task(
            "",
            action=f"[bold green]{ref_rttm_path.name}[/bold green]",
            total=len(providers),
        )

        ref_rttm_content: List[List[Union[str, float]]] = await _prepare_rttm_content(
            ref_rttm_path, "dataset"
        )
        ref_rttm: List[
            Tuple[str, float, float]
        ] = await _prepare_provider_rttm_segments(
            rttm_content=ref_rttm_content,
            target_name=dataset,
            target_type="dataset",
        )

        error = None
        scores: Dict[str, float] = {}
        for provider in providers:
            provider_rttm_path = AsyncPath(
                transcription_dir / split / provider / "rttm" / ref_rttm_path.name
            )

            if await provider_rttm_path.exists():
                hyp_rttm_content = await _prepare_rttm_content(
                    provider_rttm_path, "provider"
                )
                hyp_rttm = await _prepare_provider_rttm_segments(
                    rttm_content=hyp_rttm_content,
                    target_name=provider,
                    target_type="provider",
                )

                _score = spyder.DER(ref_rttm, hyp_rttm, collar=0.0, regions="single")
                score = ProviderComputeScore(
                    der=_score.der,
                    confusion=_score.conf,
                    miss=_score.miss,
                    false_alarm=_score.falarm,
                    status=EvaluationStatus.EVALUATED,
                )

            else:
                score = ProviderComputeScore(
                    der=None,
                    confusion=None,
                    miss=None,
                    false_alarm=None,
                    status=EvaluationStatus.NOT_FOUND,
                )

            scores[provider] = score

            step_progress.advance(step_progress_task_id)

    except Exception as e:
        scores = None
        error = f"{e}\n{traceback.format_exc()}"

    finally:
        step_progress.update(step_progress_task_id, visible=False)

    return ComputeScores(
        filename=ref_rttm_path.name,
        scores=scores,
        error=error,
    )


async def _prepare_rttm_content(
    rttm_path: Union[str, Path, AsyncPath],
    target_type: Literal["dataset", "provider"],
) -> List[List[Union[str, float]]]:
    """
    Prepare the RTTM content for evaluation. The RTTM content is a list of
    segments. We only keep the speaker, start and end time of each segment.

    Args:
        rttm_path (Union[str, Path, AsyncPath]):
            Path to the RTTM file.
        target_type (Literal["dataset", "provider"]):
            Type of the target (dataset or provider).

    Returns:
        List of RTTM segments (speaker, start, end).
    """
    if isinstance(rttm_path, (str, Path)):
        rttm_path = AsyncPath(rttm_path)

    async with aiofiles.open(rttm_path, mode="r") as file:
        content: List[str] = (await file.read()).splitlines()

    if target_type == "dataset":
        rttm_content = await _iter_dataset_rttm(content)
    elif target_type == "provider":
        rttm_content = await _iter_provider_rttm(content)

    return rttm_content


async def _iter_dataset_rttm(raw_content: List[str]) -> List[List[Union[str, float]]]:
    """Iterate over the RTTM content of a dataset."""
    rttm_content: List[List[Union[str, float]]] = []

    for line in raw_content:
        items = line.split()
        rttm_content.append(
            [str(items[7]), float(items[3]), float(items[3]) + float(items[4])]
        )

    return rttm_content


async def _iter_provider_rttm(raw_content: List[str]) -> List[List[Union[str, float]]]:
    """Iterate over the RTTM content of a provider."""
    rttm_content: List[List[Union[str, float]]] = []

    for line in raw_content:
        items = line.split()
        rttm_content.append([str(items[2]), float(items[0]), float(items[1])])

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
        (speaker_map.from_value(item[0]), item[1], item[2]) for item in rttm_content
    ]

    return prepared_ref_rttm


async def _store_evaluation_results(
    results: ProviderComputeScore,
    save_path: AsyncPath,
) -> None:
    """Store the DER results in a JSON file."""
    data = results.model_dump()
    data.pop("status")

    await save_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(save_path, mode="w") as file:
        await file.write(json.dumps(data, indent=4, ensure_ascii=False))


async def _check_cache(
    rttm_file: str,
    evaluation_dir: AsyncPath,
    split: str,
    provider: str,
) -> Tuple[bool, AsyncPath]:
    """Check the cache for the results of the diarization evaluation.

    This method check if the provider has already been evaluated for the given
    audio file. If so, it returns True, otherwise it returns False.

    Args:
        rttm_file (Path):
            The rttm file to check.
        evaluation_dir (Path):
            The evaluation directory where the results are saved, i.e. the cache.
        split (str):
            The split of the dataset.
        provider (str):
            The provider to check.

    Returns:
        Tuple[bool, Path]:
            A tuple containing a boolean indicating whether the provider has
            already been evaluated for the given audio file and the path to the
            evaluation results.
    """
    _file_name = rttm_file.split(".")[0]
    eval_output_file_path = AsyncPath(
        evaluation_dir / split / provider / f"{_file_name}.json"
    )

    eval_output_exists = await eval_output_file_path.exists()

    return eval_output_exists, eval_output_file_path
