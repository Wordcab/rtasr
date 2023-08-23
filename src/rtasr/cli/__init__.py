"""The CLI module regroups all the CLI related classes and functions."""

from .download_command import DownloadDatasetCommand
from .evaluation_command import EvaluationCommand
from .list_command import ListItemsCommand
from .transcription_command import TranscriptionASRCommand

__all__ = [
    "DownloadDatasetCommand",
    "EvaluationCommand",
    "ListItemsCommand",
    "TranscriptionASRCommand",
]
