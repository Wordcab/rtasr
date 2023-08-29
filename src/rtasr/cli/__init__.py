"""The CLI module regroups all the CLI related classes and functions."""

from .audio_length_command import AudioLengthCommand
from .download_command import DownloadDatasetCommand
from .evaluation_command import EvaluationCommand
from .list_command import ListItemsCommand
from .plot_command import PlotCommand
from .transcription_command import TranscriptionASRCommand

__all__ = [
    "AudioLengthCommand",
    "DownloadDatasetCommand",
    "EvaluationCommand",
    "ListItemsCommand",
    "PlotCommand",
    "TranscriptionASRCommand",
]
