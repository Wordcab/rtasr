"""The benchmark command."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Union

from rich import print

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, PROVIDERS
from rtasr.utils import get_api_key, resolve_cache_dir


def benchmark_asr_command_factory(args: argparse.Namespace):
    return BenchmarkASRCommand(
        args.provider,
        args.dataset,
        args.split,
        args.dataset_dir,
        args.output_dir,
        args.no_cache,
    )


class BenchmarkASRCommand:
    """Benchmark an ASR provider against a dataset."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        subparser = parser.add_parser(
            "benchmark", help="Benchmark an ASR provider against a dataset."
        )
        subparser.add_argument(
            "-p",
            "--provider",
            help="The ASR provider to benchmark.",
            required=True,
            type=str,
        )
        subparser.add_argument(
            "-d", "--dataset", help="The dataset to use.", required=True, type=str
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
        subparser.add_argument(
            "-o",
            "--output_dir",
            help=(
                "Path where store the benchmark outputs. Defaults to"
                " `~/.cache/rtasr/benchmark`."
            ),
            required=False,
            default=None,
            type=str,
        )
        subparser.add_argument(
            "--no-cache",
            help="Whether to use the cache or not.",
            required=False,
            action="store_false",
        )
        subparser.set_defaults(func=benchmark_asr_command_factory)

    def __init__(
        self,
        provider: str,
        dataset: str,
        split: str,
        dataset_dir: Union[str, None] = None,
        output_dir: Union[str, None] = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the command."""
        self.provider = provider
        self.dataset = dataset
        self.split = split
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.use_cache = use_cache

    def run(self) -> None:
        """Run the command.

        Raises:
            Exception: If the command fails.

        """
        try:
            if self.provider.lower() not in PROVIDERS.keys():
                print(
                    error_message.format(
                        input_type="provider", user_input=self.provider
                    )
                )
                print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS.keys()]))
                exit(1)

            if self.dataset.lower() not in DATASETS.keys():
                print(
                    error_message.format(input_type="dataset", user_input=self.dataset)
                )
                print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS.keys()]))
                exit(1)

            _provider = self.provider.lower()
            _dataset = self.dataset.lower()
            api_key = get_api_key(_provider)

            print(
                rf"Provider [bold green]\[{_provider}][/bold green]: API key found 🎉."
            )

            if self.dataset_dir is None:
                dataset_dir = resolve_cache_dir() / "datasets" / _dataset
            else:
                dataset_dir = Path(self.dataset_dir) / "datasets" / _dataset

            if not dataset_dir.exists():
                print(
                    f"Dataset directory does not exist: {dataset_dir.resolve()}\nPlease"
                    f" run `rtasr download -d {_dataset} --no-cache` to download the"
                    " dataset."
                )
                exit(1)

            print(
                rf"Dataset [bold green]\[{_dataset}][/bold green] from"
                f" {dataset_dir.resolve()}"
            )

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

            print(f"Using splits: {splits}")

            all_manifest_filepaths = DATASETS[_dataset]["manifest_filepaths"]

            selected_manifest_filepaths: List[Path] = []
            for split in splits:
                for filepath in all_manifest_filepaths[split]:
                    manifest_filepath = dataset_dir / filepath

                    if not manifest_filepath.exists():
                        print(
                            "Manifest file does not exist:"
                            f" {manifest_filepath.resolve()}\nPlease run `rtasr"
                            f" download -d {_dataset} --no-cache` to download the"
                            " dataset."
                        )
                        exit(1)
                    else:
                        selected_manifest_filepaths.append(manifest_filepath)

            print(
                f"Manifest filepaths: {len(selected_manifest_filepaths)} files found."
            )

            audio_filepaths: List[Path] = []
            for manifest_filepath in selected_manifest_filepaths:
                with open(manifest_filepath, "r") as f:
                    manifest = json.load(f)

                audio_filepaths.extend([Path(m["audio_filepath"]) for m in manifest])

            verified_audio_filepaths: List[Path] = [
                filepath for filepath in audio_filepaths if filepath.exists()
            ]

            print(
                f"Audio files: [bold green]{len(verified_audio_filepaths)}[/bold"
                f" green]/{len(audio_filepaths)} (found/total)"
            )

            if self.output_dir is None:
                output_dir = resolve_cache_dir() / "benchmark"
            else:
                output_dir = Path(self.output_dir) / "benchmark"

            benchmark_dir = output_dir / _dataset
            if not benchmark_dir.exists():
                benchmark_dir.mkdir(parents=True, exist_ok=True)

            if _provider == "assemblyai":
                from rtasr.asr import AssemblyAI

                engine = AssemblyAI(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )
                print("AssemblyAI is not supported yet.")
                exit(1)

            elif _provider == "aws":
                from rtasr.asr import Aws

                engine = Aws(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )
                print("AWS is not supported yet.")
                exit(1)

            elif _provider == "azure":
                from rtasr.asr import Azure

                engine = Azure(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )
                print("Azure is not supported yet.")
                exit(1)

            elif _provider == "deepgram":
                from rtasr.asr import Deepgram

                engine = Deepgram(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )

            elif _provider == "google":
                from rtasr.asr import Google

                engine = Google(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )
                print("Google is not supported yet.")
                exit(1)

            elif _provider == "revai":
                from rtasr.asr import RevAI

                engine = RevAI(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )
                print("RevAI is not supported yet.")
                exit(1)

            elif _provider == "speechmatics":
                from rtasr.asr import Speechmatics

                engine = Speechmatics(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )
                print("Speechmatics is not supported yet.")
                exit(1)

            elif _provider == "wordcab":
                from rtasr.asr import Wordcab

                engine = Wordcab(
                    api_url=PROVIDERS[_provider]["url"],
                    api_key=api_key,
                    options=PROVIDERS[_provider]["options"],
                )

            else:
                print(f"Unknown provider: {_provider}")
                exit(1)

            asyncio.run(
                engine.launch(
                    audio_files=verified_audio_filepaths[:1],
                    output_dir=benchmark_dir,
                )
            )

        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)
        except Exception as e:
            raise Exception(e) from e
