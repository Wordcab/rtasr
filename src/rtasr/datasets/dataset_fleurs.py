"""Download the google/fleurs dataset and prepare it for benchmarking.

More information about the dataset can be found here:
https://huggingface.co/datasets/google/fleurs

We use only the `en_us` subset of the dataset.
"""

import multiprocessing
import unittest.mock as mock
from pathlib import Path
from typing import Any, Dict

from datasets import DownloadMode, load_dataset
from datasets.utils.logging import tqdm
from rich import print
from tqdm.rich import tqdm as rich_tqdm

from rtasr.constants import DATASETS
from rtasr.utils import resolve_cache_dir


class rich_tqdm_cls:
    """Monkey patch the tqdm progress bar to use rich."""

    def __call__(self, *args, disable=False, **kwargs):
        """Return a tqdm instance."""
        return rich_tqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        """Set the lock for the tqdm progress bar."""
        self._lock = None
        tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        """Get the lock for the tqdm progress bar."""
        tqdm.get_lock()

    def __delattr__(self, attr):
        """fix for https://github.com/huggingface/datasets/issues/6066"""
        try:
            del self.__dict__[attr]
        except KeyError as e:
            if attr != "_lock":
                raise AttributeError(attr) from e

tqdm = rich_tqdm_cls()  # noqa: F811


def prepare_fleurs_dataset(output_dir: str = None, use_cache: bool = True) -> None:
    """Prepare the google/fleurs dataset."""
    if output_dir is None:
        cache_datasets_dir = resolve_cache_dir() / "datasets"
        cache_datasets_dir.mkdir(parents=True, exist_ok=True)
        output_dir = cache_datasets_dir / "fleurs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_cache:
        download_mode = DownloadMode.REUSE_DATASET_IF_EXISTS
    else:
        download_mode = DownloadMode.FORCE_REDOWNLOAD

    dataset_metadata: Dict[str, Any] = DATASETS["fleurs"]

    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPUs for downloading the dataset.")

    for split in dataset_metadata["splits"]:
        load_dataset(
            "google/fleurs", "en_us",
            split=split,
            cache_dir=str(output_dir),
            num_proc=num_cpus,
            download_mode=download_mode,
        )

        print(f"✅ Downloaded {split} split of the dataset.")
