"""The CLI module regroups all the CLI related classes and functions."""

from .benchmark_command import BenchmarkASRCommand
from .download_command import DownloadDatasetCommand
from .list_command import ListItemsCommand

__all__ = [
    "BenchmarkASRCommand",
    "DownloadDatasetCommand",
    "ListItemsCommand",
]
