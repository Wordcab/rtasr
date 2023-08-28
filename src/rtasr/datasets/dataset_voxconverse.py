"""Download the VoxConverse dataset and prepare it for benchmarking.

More information here: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/

This dataset is used for Speaker Diarization.

Usage:
    rtasr download -d voxconverse --no-cache
"""

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import aiohttp
from rich import print
from rich.live import Live

from rtasr.constants import DATASETS
from rtasr.utils import (
    create_live_panel,
    download_file,
    get_files,
    resolve_cache_dir,
    unzip_file,
)


async def prepare_voxconverse_dataset(
    output_dir: str = None, use_cache: bool = True
) -> None:
    """
    Download the VoxConverse dataset and prepare it for benchmarking.

    Args:
        output_dir (str):
            Path to the directory where the dataset is stored.
            If not specified, the dataset will be stored in the
            default `~/.cache/rtasr/datasets` directory.
        use_cache (bool):
            Whether to use the cache or not. Defaults to True.
            If set to False, the dataset will be downloaded
            again even if it is already present in the cache.
    """
    if output_dir is None:
        cache_datasets_dir = resolve_cache_dir() / "datasets"
        cache_datasets_dir.mkdir(parents=True, exist_ok=True)
        output_dir = cache_datasets_dir / "voxconverse"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata: Dict[str, Any] = DATASETS["voxconverse"]

    current_progress, _, splits_progress, progress_group = create_live_panel()

    manifest_split_paths: List[Path] = []
    with Live(progress_group):
        async with aiohttp.ClientSession() as session:
            current_progress_task_id = current_progress.add_task(
                "Downloading dataset files: VoxConverse"
            )
            splits_progress_task_id = splits_progress.add_task(
                "", total=len(dataset_metadata["zip_urls"])
            )

            zip_tasks = []
            for zip_name, zip_url in dataset_metadata["zip_urls"].items():
                zip_dir = output_dir / zip_name
                zip_dir.mkdir(parents=True, exist_ok=True)

                zip_tasks.append(_download_zip(zip_url, zip_dir, session, use_cache))

            zip_filepaths: List[Path] = []
            for future in asyncio.as_completed(zip_tasks):
                try:
                    zip_filepath = await future
                    if not isinstance(zip_filepath, Path):
                        raise Exception(
                            "Download failed for zip:"
                            f" {zip_filepath}\n{traceback.format_exc()}"
                        )
                    zip_filepaths.append(zip_filepath)
                except Exception as e:
                    raise Exception(e) from e
                finally:
                    splits_progress.advance(splits_progress_task_id)

            splits_progress.update(splits_progress_task_id, visible=False)

            current_progress.stop_task(current_progress_task_id)
            current_progress.update(
                current_progress_task_id,
                description="[bold green]Dataset VoxConverse zip files downloaded.",
            )

        current_progress_task_id = current_progress.add_task(
            "Unzipping files: VoxConverse"
        )
        splits_progress_task_id = splits_progress.add_task("", total=len(zip_filepaths))

        unzipping_tasks = []
        for zip_filepath in zip_filepaths:
            zip_dir = zip_filepath.parent
            unzipping_tasks.append(unzip_file(zip_filepath, zip_dir, use_cache))

        for future in asyncio.as_completed(unzipping_tasks):
            try:
                unzipped_dir = await future
                if not isinstance(unzipped_dir, Path):
                    raise Exception(
                        "Unzipping failed for zip:"
                        f" {unzipped_dir}\n{traceback.format_exc()}"
                    )
            except Exception as e:
                raise Exception(e) from e
            finally:
                splits_progress.advance(splits_progress_task_id)

        splits_progress.update(splits_progress_task_id, visible=False)

        current_progress.stop_task(current_progress_task_id)
        current_progress.update(
            current_progress_task_id,
            description="[bold green]Dataset VoxConverse files unzipped.",
        )

        current_progress_task_id = current_progress.add_task(
            "Preparing manifest files: VoxConverse"
        )
        splits_progress_task_id = splits_progress.add_task("", total=2)

        split_tasks = []
        for split in dataset_metadata["splits"]:
            split_audio = output_dir / dataset_metadata["filepaths"][split]
            split_rttm = output_dir / dataset_metadata["filepaths"]["rttm"] / split
            split_tasks.append(
                _prepare_voxconverse_manifest_split(
                    split,
                    split_audio,
                    split_rttm,
                    output_dir,
                    use_cache,
                )
            )

        for future in asyncio.as_completed(split_tasks):
            try:
                manifest_path = await future
                if not isinstance(manifest_path, Path):
                    raise Exception(
                        "Manifest preparation failed for one split:"
                        f" {manifest_path}\n{traceback.format_exc()}"
                    )
                else:
                    manifest_split_paths.append(manifest_path)
            except Exception as e:
                raise Exception(e) from e
            finally:
                splits_progress.advance(splits_progress_task_id)

        splits_progress.update(splits_progress_task_id, visible=False)
        current_progress.stop_task(current_progress_task_id)
        current_progress.update(
            current_progress_task_id,
            description="[bold green]Manifest files prepared.",
        )

    print("[bold green]Manifest files created:[/bold green]")
    for manifest_path in manifest_split_paths:
        print(f"  - {manifest_path}")


async def _download_zip(
    zip_url: str,
    zip_dir: Path,
    session: aiohttp.ClientSession,
    use_cache: bool,
) -> Path:
    """Download a zip file from the VoxConverse dataset."""
    filepath = await download_file(zip_url, zip_dir, session, use_cache)

    return filepath


async def _prepare_voxconverse_manifest_split(
    split: str,
    split_audio: Path,
    split_rttm: Path,
    output_dir: Path,
    use_cache: bool,
) -> List[Path]:
    """Prepare a manifest file."""
    rttm_files = []
    for path in get_files(split_rttm):
        rttm_files.append(path)

    audio_files = []
    for path in get_files(split_audio):
        audio_files.append(path)

    manifest_path = output_dir / f"{split}_manifest.json"
    if use_cache is False or not manifest_path.exists():
        manifest_path = await _create_manifest(audio_files, manifest_path, rttm_files)

    return manifest_path


async def _create_manifest(
    audio_files: List[Path],
    manifest_filepath: Path,
    rttm_files: List[Path],
) -> Path:
    """Create a manifest file."""
    if not len(audio_files) == len(rttm_files):
        raise Exception(
            f"Number of audio files ({len(audio_files)}), rttm files"
            f" ({len(rttm_files)}) do not match."
        )

    if manifest_filepath.exists():
        manifest_filepath.unlink(missing_ok=True)

    manifest_file_content = []
    for audio, rttm in zip(sorted(audio_files), sorted(rttm_files)):
        assert audio.stem.split(".")[0] == rttm.stem.split(".")[0]

        async with aiofiles.open(rttm, mode="r") as f:
            content = await f.read()

        labels = [
            line.strip().split()[7] for line in content.splitlines() if line.strip()
        ]
        meta = [
            {
                "audio_filepath": str(audio),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": len(set(labels)),
                "rttm_filepath": str(rttm),
                "uem_filepath": None,
                "ctm_filepath": None,
            }
        ]
        manifest_file_content.extend(meta)

    async with aiofiles.open(manifest_filepath, mode="w") as f:
        await f.write(json.dumps(manifest_file_content, indent=4))

    return manifest_filepath
