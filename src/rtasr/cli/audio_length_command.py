"""The audio-length command."""

import argparse
import os
from pathlib import Path
from typing import List

from datasets import load_dataset
from rich import print

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, PROVIDERS
from rtasr.utils import (
    get_audio_duration_from_file,
    get_audio_duration_from_samples,
    get_human_readable_duration,
    get_human_readable_price,
)


def audio_length_command_factory(args: argparse.Namespace):
    return AudioLengthCommand(
        dataset=args.dataset,
        split=args.split,
        dataset_dir=args.dataset_dir,
    )


class AudioLengthCommand:
    """Audio length command."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        """Register the subcommand."""
        subparser = parser.add_parser("audio-length", help="Audio length command.")
        subparser.add_argument(
            "-d",
            "--dataset",
            help="The dataset to plot.",
            type=str,
            required=True,
        )
        subparser.add_argument(
            "-s",
            "--split",
            help="The split to use. `rtasr list -t datasets` for more info.",
            required=False,
            default="all",
            type=str,
        )
        subparser.add_argument(
            "--dataset_dir",
            help=(
                "Path where the dataset files are stored. Defaults to"
                " `~/.cache/rtasr/datasets`."
            ),
            required=False,
            default=None,
            type=str,
        )
        subparser.set_defaults(func=audio_length_command_factory)

    def __init__(
        self,
        dataset: str,
        split: str,
        dataset_dir: str,
    ) -> None:
        """Initialize the command."""
        self.dataset = dataset
        self.split = split
        self.dataset_dir = dataset_dir

    def run(self) -> None:
        """Run the command."""
        try:
            if self.dataset.lower() not in DATASETS.keys():
                print(
                    error_message.format(input_type="dataset", user_input=self.dataset)
                )
                print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS.keys()]))
                exit(1)

            _dataset = self.dataset.lower()

            if self.dataset_dir is None:
                dataset_dir = Path.home() / ".cache" / "rtasr" / "datasets"
            else:
                dataset_dir = Path(self.dataset_dir)

            splits: List[str] = []
            if self.split.lower() == "all":
                splits.extend(DATASETS[self.dataset.lower()]["splits"])
            elif self.split.lower() in DATASETS[self.dataset.lower()]["splits"]:
                splits.append(self.split.lower())
            else:
                print(error_message.format(input_type="split", user_input=self.split))
                print(
                    "".join(
                        [
                            f"  - [bold]{s}[bold]\n"
                            for s in DATASETS[self.dataset.lower()]["splits"]
                        ]
                    )
                )
                exit(1)

            audio_durations = []
            for split in splits:
                if _dataset == "fleurs":
                    os.environ["HF_DATASETS_OFFLINE"] = "1"
                    try:
                        hf_dataset = load_dataset(
                            "google/fleurs",
                            "en_us",
                            cache_dir=str(dataset_dir / "fleurs"),
                        )
                    except Exception:
                        print(
                            "Failed to load dataset from"
                            f" {dataset_dir.resolve()}.\nPlease run `rtasr download -d"
                            " fleurs --no-cache` to download the dataset."
                        )
                        exit(1)

                    for audio_file in hf_dataset[split]:
                        audio_durations.append(
                            get_audio_duration_from_samples(
                                samples=audio_file["num_samples"],
                                sample_rate=audio_file["audio"]["sampling_rate"],
                            )
                        )

                else:
                    split_dir = dataset_dir / _dataset / split / "audio"
                    for audio_file in split_dir.glob("**/*.wav"):
                        audio_durations.append(get_audio_duration_from_file(audio_file))

            if len(audio_durations) == 0:
                print("[bold red]No audio files found.[/bold red]")
                exit(1)

            print(
                "[bold green]Total audio length for"
                f" {_dataset.upper()} ({splits}):[/bold green]\n[bold"
                f" yellow]{get_human_readable_duration(audio_durations)} (minutes:seconds)"
                " [/bold yellow]"
            )
            print("Estimated pricing:")
            for _, val in PROVIDERS.items():
                if val["pricing"] != {}:
                    print(
                        f"[bold purple]  - {val['engine']}:[/bold purple]"
                        f" {get_human_readable_price(audio_durations, val['pricing'])}"
                    )

        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)

        except Exception as e:
            raise Exception(e) from e
