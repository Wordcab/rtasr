"""Word Error Rate (WER) evaluation implementation."""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from aiopath import AsyncPath
from jiwer import process_words
from jiwer.transformations import wer_contiguous
from pydantic import BaseModel
from rich.progress import Progress, TaskID

from rtasr.constants import PROVIDERS
from rtasr.evaluation.schemas import (
    EvaluationResult,
    EvaluationStatus,
    ProviderResult,
    TaskStatus,
)


class ProviderComputeScore(BaseModel):
    """The score computed for a provider."""

    hits: Union[int, None]
    mer: Union[float, None]
    wer: Union[float, None]
    wil: Union[float, None]
    status: EvaluationStatus


class ComputeScores(BaseModel):
    """The result of a compute score task."""

    error: Union[str, None]
    filename: str
    scores: Union[Dict[str, ProviderComputeScore], None]


async def evaluate_wer(
    dataset: str,
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
                dataset=dataset,
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


async def compute_score(
    dataset: str,
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
        ref_dialogue_content: List[List[Union[str, float]]] = await _prepare_dialogue_content(
            ref_dialogue_path, "dataset"
        )
        ref_dialogue: List[
            Tuple[str, float, float]
        ] = await _prepare_dialogue_str(
            rttm_content=ref_dialogue_content,
            target_name=dataset,
            target_type="dataset",
        )

        scores: Dict[str, float] = {}
        for provider in providers:
            current_provider = provider
            provider_rttm_path = AsyncPath(
                transcription_dir / split / provider / "dialogue" / ref_dialogue_path.name
            )

            if await provider_rttm_path.exists():
                hyp_dialogue_content = await _prepare_dialogue_content(
                    provider_rttm_path, "provider"
                )
                hyp_dialogue = await _prepare_dialogue_str(
                    rttm_content=hyp_dialogue_content,
                    target_name=provider,
                    target_type="provider",
                )

                _score = process_words(
                    reference=ref_dialogue,
                    hypothesis=hyp_dialogue,
                    reference_transform=wer_contiguous,
                    hypothesis_transform=wer_contiguous,
                )
                score = ProviderComputeScore(
                    hits=_score.hits,
                    mer=_score.mer,
                    wer=_score.wer,
                    wil=_score.wil,
                    status=EvaluationStatus.EVALUATED,
                )

            else:
                score = ProviderComputeScore(
                    hits=None,
                    mer=None,
                    wer=None,
                    wil=None,
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


async def _prepare_dialogue_content() -> List[str]:
    """"""
    pass

async def _prepare_dialogue_str() -> str:
    """"""
    pass
