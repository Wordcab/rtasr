"""Define all the constants used in rtasr."""

from typing import OrderedDict, Tuple

from rich.console import Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


DATASETS = OrderedDict(
    [
        (
            "ami",
            {
                "splits": ["test", "dev", "train"],
                "audio_types": ["Mix-Headset", "Array1-01"],
                "urls": {
                    "rttm": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm",
                    "uem": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem",
                    "list": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt",
                },
                "exclude_ids": ["IS1007d", "IS1003b"],
                "manifest_filepaths": {
                    "test": [
                        "test/manifest_Array1-01.json",
                        "test/manifest_Mix-Headset.json",
                    ],
                    "dev": [
                        "dev/manifest_Array1-01.json",
                        "dev/manifest_Mix-Headset.json",
                    ],
                    "train": [
                        "train/manifest_Array1-01.json",
                        "train/manifest_Mix-Headset.json",
                    ],
                },
            },
        ),
        (
            "voxconverse",
            {
                "splits": ["dev", "test"],
                "zip_urls": {
                    "rttm": "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip",
                    "dev": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
                    "test": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip",
                },
                "filepaths": {
                    "dev": "dev/audio",
                    "test": "test/voxconverse_test_wav",
                    "rttm": "rttm/voxconverse-master",
                },
                "manifest_filepaths": {
                    "dev": ["dev_manifest.json"],
                    "test": ["test_manifest.json"],
                },
            },
        ),
    ]
)
PROVIDERS = OrderedDict(
    [
        (
            "assemblyai",
            {
                "url": "",
            },
        ),
        (
            "aws",
            {
                "url": "",
            },
        ),
        (
            "azure",
            {
                "url": "",
            },
        ),
        (
            "deepgram",
            {
                "url": "https://api.deepgram.com/v1/listen",
                "options": {
                    "diarize": True,
                    "model": "nova",
                    "punctuate": True,
                    "utterances": True,
                },
            },
        ),
        (
            "google",
            {
                "url": "",
            },
        ),
        (
            "revai",
            {
                "url": "",
            },
        ),
        (
            "speechmatics",
            {
                "url": "",
            },
        ),
        (
            "wordcab",
            {
                "url": "https://wordcab.com/api/v1/transcribe",
            },
        ),
    ]
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
