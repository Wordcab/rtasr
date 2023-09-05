"""Word Error Rate (WER) evaluation implementation."""

import asyncio
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import aiofiles
from aiopath import AsyncPath
from jiwer import process_words
from jiwer.transformations import wer_contiguous
from pydantic import BaseModel
from rich.progress import Progress, TaskID
from typing_extensions import Literal

from rtasr.constants import PROVIDERS
from rtasr.evaluation.schemas import (
    EvaluationResult,
    EvaluationStatus,
    ProviderResult,
    TaskStatus,
)
from rtasr.utils import (
    _check_cache,
    attach_punctuation_to_last_word,
    reconstruct_acronym,
    remove_bracketed_text,
    store_evaluation_results,
)


class ProviderComputeScore(BaseModel):
    """The score computed for a provider."""

    hits: Union[int, None]
    mer: Union[float, None]
    wer: Union[float, None]
    wil: Union[float, None]
    wrr: Union[float, None]
    status: EvaluationStatus


class ComputeScores(BaseModel):
    """The result of a compute score task."""

    error: Union[str, None]
    filename: str
    scores: Union[Dict[str, ProviderComputeScore], None]


async def evaluate_wer(
    split_name: str,
    split_dialogue_files: List[Path],
    evaluation_dir: Path,
    transcription_dir: Path,
    split_progress: Progress,
    split_progress_task_id: TaskID,
    step_progress: Progress,
    use_cache: bool,
    debug: bool,
) -> EvaluationResult:
    """"""
    if debug:
        split_dialogue_files = split_dialogue_files[:1]

    step_progress_task_id = step_progress.add_task(
        "",
        action=f"[bold green]{split_name}[/bold green]",
        total=len(split_dialogue_files),
    )

    providers = list(PROVIDERS.keys())
    task_tracking: Dict[str, Any] = {}
    tasks: List[Callable] = []

    for dialogue_file in split_dialogue_files:
        task_tracking[dialogue_file.name] = {
            "dialogue_file_name": dialogue_file.name,
            "status": TaskStatus.IN_PROGRESS,
            "provider_results": {provider: None for provider in providers},
        }
        tasks.append(
            compute_score(
                ref_dialogue_path=dialogue_file,
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
            task_tracking[filename]["status"] = TaskStatus.ERROR
            task_tracking[filename]["error"] = f"{filename} -> {task_result.error}"

        else:
            task_tracking[filename]["status"] = TaskStatus.DONE

            for provider in task_result.scores:
                status = task_result.scores[provider].status

                if (
                    status == EvaluationStatus.CACHED
                    or status == EvaluationStatus.EVALUATED
                ):
                    _results = task_result.scores[provider].model_dump()
                    wrr_results = _results.pop("wrr")

                    # WER
                    wer_file_exists, wer_file_path = await _check_cache(
                        file_name=filename,
                        evaluation_dir=evaluation_dir,
                        split=split_name,
                        provider=provider,
                        metric="wer",
                    )

                    if use_cache and wer_file_exists:
                        task_tracking[filename]["provider_results"][
                            provider
                        ] = EvaluationStatus.CACHED
                    else:
                        task_tracking[filename]["provider_results"][
                            provider
                        ] = EvaluationStatus.EVALUATED
                        await store_evaluation_results(
                            results=_results,
                            save_path=wer_file_path,
                        )

                    # WRR
                    wrr_file_exists, wrr_file_path = await _check_cache(
                        file_name=filename,
                        evaluation_dir=evaluation_dir,
                        split=split_name,
                        provider=provider,
                        metric="wrr",
                    )

                    if use_cache and wrr_file_exists:
                        task_tracking[filename]["provider_results"][
                            provider
                        ] = EvaluationStatus.CACHED
                    else:
                        task_tracking[filename]["provider_results"][
                            provider
                        ] = EvaluationStatus.EVALUATED
                        await store_evaluation_results(
                            results={"wrr": wrr_results, "status": status},
                            save_path=wrr_file_path,
                        )

                else:
                    task_tracking[filename]["provider_results"][
                        provider
                    ] = EvaluationStatus.NOT_FOUND

        step_progress.advance(step_progress_task_id)

    step_progress.update(step_progress_task_id, advance=len(split_dialogue_files))
    split_progress.advance(split_progress_task_id)

    results: List[ProviderResult] = []
    for provider in providers:
        counter = Counter(
            [
                task_tracking[dialogue_file.name]["provider_results"][provider]
                for dialogue_file in split_dialogue_files
            ]
        )
        results.append(
            ProviderResult(
                cached=counter[EvaluationStatus.CACHED],
                evaluated=counter[EvaluationStatus.EVALUATED],
                not_found=counter[EvaluationStatus.NOT_FOUND],
                provider_name=provider,
            )
        )

    errors: List[str] = [
        task_tracking[dialogue_file.name]["error"]
        for dialogue_file in split_dialogue_files
        if task_tracking[dialogue_file.name]["status"] == TaskStatus.ERROR
    ]

    return EvaluationResult(
        errors=errors,
        split_name=split_name,
        results=results,
    )


async def compute_score(
    ref_dialogue_path: AsyncPath,
    providers: List[str],
    split: str,
    transcription_dir: Path,
    step_progress: Progress,
    use_cache: bool,
) -> ComputeScores:
    """"""
    step_progress_task_id = step_progress.add_task(
        "",
        action=f"[bold green]{ref_dialogue_path.name}[/bold green]",
        total=len(providers),
    )
    current_provider = None  # Error tracking purpose
    error = None

    try:
        ref_dialogue: List[str] = await _prepare_dialogue_content(
            ref_dialogue_path, "dataset"
        )

        scores: Dict[str, float] = {}
        for provider in providers:
            current_provider = provider
            provider_dialogue_path = AsyncPath(
                transcription_dir
                / split
                / provider
                / "dialogue"
                / f"{ref_dialogue_path.stem}.txt"
            )

            if await provider_dialogue_path.exists():
                hyp_dialogue: List[str] = await _prepare_dialogue_content(
                    provider_dialogue_path, "provider"
                )

                _score = process_words(
                    reference=ref_dialogue,
                    hypothesis=hyp_dialogue,
                    reference_transform=wer_contiguous,
                    hypothesis_transform=wer_contiguous,
                )
                _wrr = 1 - _score.wer
                score = ProviderComputeScore(
                    hits=_score.hits,
                    mer=_score.mer,
                    wer=_score.wer,
                    wil=_score.wil,
                    wrr=_wrr,
                    status=EvaluationStatus.EVALUATED,
                )

            else:
                score = ProviderComputeScore(
                    hits=None,
                    mer=None,
                    wer=None,
                    wil=None,
                    wrr=None,
                    status=EvaluationStatus.NOT_FOUND,
                )

            scores[provider] = score

            step_progress.advance(step_progress_task_id)

    except Exception:
        scores = None
        error_type, error_instance, _ = sys.exc_info()
        error = (
            f"[bold]({current_provider})[/bold] "
            f"{traceback.format_exception_only(error_type, error_instance)[-1].strip()}"
        )

    finally:
        step_progress.update(step_progress_task_id, visible=False)

    return ComputeScores(
        filename=ref_dialogue_path.name,
        scores=scores,
        error=error,
    )


async def _prepare_dialogue_content(
    dialogue_path: Union[str, Path, AsyncPath],
    target_type: Literal["dataset", "provider"],
) -> List[str]:
    """
    Prepare the dialogue content for evaluation.

    Args:
        dialogue_path (Union[str, Path, AsyncPath]):
            Path to the dialogue file.
        target_type (Literal["dataset", "provider"]):
            Type of the target. Either "dataset" or "provider".

    Returns:
        List[str]: The dialogue content as a list of strings.
    """
    if isinstance(dialogue_path, (str, Path)):
        dialogue_path = AsyncPath(dialogue_path)

    if target_type == "dataset":
        async with aiofiles.open(dialogue_path, mode="r") as file:
            content: str = await file.read()

        if dialogue_path.suffix == ".json":
            data: dict = json.loads(content)
            dialogue_content = _format_dialogue_content(dialogue_content=data)
        else:
            dialogue_content = content.splitlines()

    elif target_type == "provider":
        async with aiofiles.open(dialogue_path, mode="r") as file:
            dialogue_content: List[str] = (await file.read()).splitlines()

    return dialogue_content


def _format_dialogue_content(dialogue_content: dict) -> List[str]:
    """
    Format the dialogue content to a list of strings.

    Args:
        dialogue_content (dict):
            The dialogue content as a dictionary.

    Returns:
        List[str]: The dialogue content as a list of strings.
    """
    current_speaker: Union[str, None] = None
    current_sentence: str = ""

    formatted_dialogue: List[str] = []
    for utterance in dialogue_content:
        text = remove_bracketed_text(utterance["text"])
        text = reconstruct_acronym(text)
        text = attach_punctuation_to_last_word(text)

        if text == "":
            continue

        if current_speaker != utterance["speaker"] and current_speaker is not None:
            formatted_dialogue.append(current_sentence)
            current_sentence = ""
        else:
            current_sentence += " " + text

        current_speaker = utterance["speaker"]

    formatted_dialogue.append(current_sentence)

    return [sentence.strip() for sentence in formatted_dialogue if sentence != ""]
