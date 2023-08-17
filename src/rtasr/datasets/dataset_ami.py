"""Download the AMI dataset and prepare it for benchmarking.

More information here: https://groups.inf.ed.ac.uk/ami/corpus/

This dataset is used for Speaker Diarization.

Usage: TODO
"""

import asyncio
from pathlib import Path
from typing import Dict, List

import aiofiles
import aiohttp
from rich import print
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style

from rtasr.utils import download_file, resolve_cache_dir

# from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

DATASET_METADATA = {
    "splits": ["test", "dev", "train"],
    "audio_types": ["Mix-Headset", "Array1-01"],
    "urls": {
        "rttm": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm",
        "uem": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem",
        "list": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt",
    },
    "exclude_ids": ["IS1007d", "IS1003b"],
}


# Inspired from:
# https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/speaker_tasks/get_ami_data.py
async def download_ami_dataset(output_dir: str = None) -> None:
    """
    Download the AMI dataset and prepare it for benchmarking.

    Args:
        output_dir (str):
            Path to the directory where the dataset is stored.
            If not specified, the dataset will be stored in the
            default `~/.cache/rtasr/datasets` directory.
    """
    if output_dir is None:
        cache_datasets_dir = resolve_cache_dir() / "datasets"
        cache_datasets_dir.mkdir(parents=True, exist_ok=True)
        output_dir = cache_datasets_dir / "ami"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    current_progress = Progress(TimeElapsedColumn(), TextColumn("{task.description}"))
    step_progress = Progress(
        TextColumn("  "),
        TimeElapsedColumn(),
        SpinnerColumn("dots", finished_text="✅", speed=0.5),
        TextColumn("[bold purple]{task.fields[action]}"),
        BarColumn(bar_width=20),
    )
    splits_progress = Progress(
        TextColumn("[bold blue]Progres: {task.percentage:.0f}%"),
        BarColumn(),
        TextColumn("({task.completed} of {task.total} steps done)"),
    )
    progress_group = Group(Panel(Group(current_progress, step_progress, splits_progress)))

    with Live(progress_group):
        async with aiohttp.ClientSession() as session:
            current_progress_task_id = current_progress.add_task("Downloading dataset files: AMI")
            splits_progress_task_id = splits_progress.add_task("", total=len(DATASET_METADATA["splits"]))

            split_tasks = []
            for split in DATASET_METADATA["splits"]:
                split_tasks.append(_download_ami_split(split, output_dir, session, step_progress))

            for future in asyncio.as_completed(split_tasks):
                try:
                    await future
                except Exception as e:
                    print(e, Style(color="red", blink=True, bold=True))
                finally:
                    splits_progress.advance(splits_progress_task_id)

            splits_progress.update(splits_progress_task_id, visible=False)

            current_progress.stop_task(current_progress_task_id)
            current_progress.update(
                current_progress_task_id, description=f"[bold green]Dataset AMI downloaded."
            )


async def _download_ami_split(
    split: str, output_dir: Path, session: aiohttp.ClientSession, step_progress: Progress
) -> None:
    """Download a split of the AMI dataset."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    filepath = await download_file(
        DATASET_METADATA["urls"]["list"].format(split), split_dir, session
    )
    async with aiofiles.open(filepath, mode="r") as f:
        content = await f.read()

    file_ids = content.strip().split('\n')
    filtered_file_ids = [file_id for file_id in file_ids if file_id not in DATASET_METADATA["exclude_ids"]]
    audio_types = DATASET_METADATA["audio_types"]

    file_download_results: List[Dict[str, str]] = []
    for file_id in filtered_file_ids:
        for audio_type in audio_types:
            file_download_results.append(
                download_file(
                    url=f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{file_id}/audio/{file_id}.{audio_type}.wav",
                    output_dir=split_dir / "audio",
                    session=session,
                )
            )
        file_download_results.append(
            download_file(
                url=DATASET_METADATA["urls"]["rttm"].format(split, file_id),
                output_dir=split_dir / "rttm",
                session=session,
            )
        )
        file_download_results.append(
            download_file(
                url=DATASET_METADATA["urls"]["uem"].format(split, file_id),
                output_dir=split_dir / "uem",
                session=session,
            )
        )

    step_task_id = step_progress.add_task("", action=split, total=len(file_download_results))
    for future in asyncio.as_completed(file_download_results):
        try:
            await future
        except Exception as e:
            print(e, Style(color="red", blink=True, bold=True))
        finally:
            step_progress.advance(step_task_id)

#     for manifest_path, split in (
#         (args.test_manifest_filepath, 'test'),
#         (args.dev_manifest_filepath, 'dev'),
#         (args.train_manifest_filepath, 'train'),
#     ):
#         split_path = os.path.join(data_path, split)
#         audio_path = os.path.join(split_path, "audio")
#         os.makedirs(split_path, exist_ok=True)
#         rttm_path = os.path.join(split_path, "rttm")
#         uem_path = os.path.join(split_path, "uem")

#         os.system(f"wget -P {split_path} {list_url.format(split)}")
#         with open(os.path.join(split_path, f"{split}.meetings.txt")) as f:
#             ids = f.read().strip().split('\n')
#         for id in [file_id for file_id in ids if file_id not in not_found_ids]:
#             for audio_type in audio_types:
#                 audio_type_path = os.path.join(audio_path, audio_type)
#                 os.makedirs(audio_type_path, exist_ok=True)
#                 os.system(
#                     f"wget -P {audio_type_path} https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{id}/audio/{id}.{audio_type}.wav"
#                 )
#             rttm_download = rttm_url.format(split, id)
#             os.system(f"wget -P {rttm_path} {rttm_download}")
#             uem_download = uem_url.format(split, id)
#             os.system(f"wget -P {uem_path} {uem_download}")

#         rttm_files_path = os.path.join(split_path, 'rttm_files.txt')
#         with open(rttm_files_path, 'w') as f:
#             f.write('\n'.join(os.path.join(rttm_path, p) for p in os.listdir(rttm_path)))
#         uem_files_path = os.path.join(split_path, 'uem_files.txt')
#         with open(uem_files_path, 'w') as f:
#             f.write('\n'.join(os.path.join(uem_path, p) for p in os.listdir(uem_path)))
#         for audio_type in audio_types:
#             audio_type_path = os.path.join(audio_path, audio_type)
#             audio_files_path = os.path.join(split_path, f'audio_files_{audio_type}.txt')
#             with open(audio_files_path, 'w') as f:
#                 f.write('\n'.join(os.path.join(audio_type_path, p) for p in os.listdir(audio_type_path)))
#             audio_type_manifest_path = manifest_path.replace('.json', f'.{audio_type}.json')
#             create_manifest(
#                 audio_files_path, audio_type_manifest_path, rttm_path=rttm_files_path, uem_path=uem_files_path
#             )