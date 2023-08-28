"""Word Error Rate (WER) evaluation implementation."""

from pathlib import Path
from typing import Any, Callable, Dict, List

from rich.progress import Progress, TaskID

from rtasr.constants import PROVIDERS
from rtasr.evaluation.schemas import EvaluationResult, ProviderResult


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
