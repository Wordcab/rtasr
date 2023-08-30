"""Download the google/fleurs dataset and prepare it for benchmarking.

More information about the dataset can be found here:
https://huggingface.co/datasets/google/fleurs

We use only the `en_us` subset of the dataset.
"""

from pathlib import Path

from datasets import DownloadMode, load_dataset
from datasets.utils import logging
from rich import print
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.rich import FractionColumn, RateColumn
from tqdm.std import tqdm as std_tqdm

from rtasr.utils import resolve_cache_dir


# From tqdm.rich.py: https://github.com/tqdm/tqdm/blob/master/tqdm/rich.py
class tqdm_rich(std_tqdm):  # pragma: no cover
    """Experimental rich.progress GUI version of tqdm!"""

    _shared_prog = None
    console = Console()

    @classmethod
    def get_or_create_progress(cls, *progress_args, **progress_kwargs):
        """Get the shared Progress instance or create one if it doesn't exist."""
        if cls._shared_prog is None:
            cls._shared_prog = Progress(*progress_args, console=cls.console, **progress_kwargs)
            cls._shared_prog.__enter__()

        return cls._shared_prog


    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        options  : dict, optional
            keyword arguments for `rich.progress.Progress()`.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        kwargs['disable'] = bool(kwargs.get('disable', False))
        progress = kwargs.pop('progress', None)
        options = kwargs.pop('options', {}).copy()

        super(tqdm_rich, self).__init__(*args, **kwargs)

        if self.disable:
            return

        d = self.format_dict
        if progress is None:
            progress = (
                "[progress.description]{task.description}"
                "[progress.percentage]{task.percentage:>4.0f}%",
                BarColumn(bar_width=None),
                FractionColumn(
                    unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                "[", TimeElapsedColumn(), "<", TimeRemainingColumn(),
                ",", RateColumn(unit=d['unit'], unit_scale=d['unit_scale'],
                                unit_divisor=d['unit_divisor']), "]"
            )
        options.setdefault('transient', not self.leave)

        self._prog = self.get_or_create_progress(*progress, **options)
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable:
            return
        super(tqdm_rich, self).close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        if not hasattr(self, '_prog'):
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(total=total)
        super(tqdm_rich, self).reset(total=total)


class rich_tqdm_cls:
    """Monkey patch the tqdm progress bar to use rich."""

    def __call__(self, *args, disable=False, **kwargs):
        """Return a tqdm instance."""
        if kwargs:
            _total = kwargs.get("total", None)
            if _total is None:
                disable = True

        return tqdm_rich(*args, disable=disable, **kwargs)

    def set_lock(self, *args, **kwargs):
        """Set the lock for the tqdm progress bar."""
        self._lock = None
        tqdm_rich.set_lock(*args, **kwargs)

    def get_lock(self):
        """Get the lock for the tqdm progress bar."""
        tqdm_rich.get_lock()

    def __delattr__(self, attr):
        """fix for https://github.com/huggingface/datasets/issues/6066"""
        try:
            del self.__dict__[attr]
        except KeyError as e:
            if attr != "_lock":
                raise AttributeError(attr) from e

# Overwrite the tqdm reference in datasets.utils.logging
logging.tqdm = rich_tqdm_cls()


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

    load_dataset(
        "google/fleurs", "en_us",
        cache_dir=str(output_dir),
        download_mode=download_mode,
    )

    print("🌸 Downloaded all splits")
