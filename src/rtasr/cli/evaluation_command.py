"""The evaluation command."""

import argparse
from pathlib import Path
from typing import List, Union

from rich import print

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, PROVIDERS, Metrics
from rtasr.utils import resolve_cache_dir


def evaluation_command_factory(args: argparse.Namespace):
    return EvaluationCommand(
        metrics=args.metrics,
        dataset=args.dataset,
        providers=args.providers,
        split=args.split,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        no_cache=args.no_cache,
        debug=args.debug,
    )


class EvaluationCommand:
    """List items."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        """Register the subcommand."""
        subparser = parser.add_parser("evaluation", help="Run evaluation on a dataset.")
        subparser.add_argument(
            "-m",
            "--metrics",
            help="The metrics to use. `rtasr list -t metrics` for more info.",
            required=True,
            type=str,
            nargs="+",
        )
        subparser.add_argument(
            "-d",
            "--dataset",
            help="The dataset to run evaluation on.",
            required=True,
            type=str,
        )
        subparser.add_argument(
            "-p",
            "--providers",
            help="The providers to use. `rtasr list -t providers` for more info.",
            required=False,
            default="all",
            type=str,
            nargs="+",
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
                "Path where store the transcription outputs. Defaults to"
                " `~/.cache/rtasr/transcription`."
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
        subparser.add_argument(
            "--debug",
            help="Whether to run in debug mode or not.",
            required=False,
            action="store_true",
        )
        subparser.set_defaults(func=evaluation_command_factory)

    def __init__(
        self,
        metrics: List[str],
        dataset: str,
        providers: Union[str, List[str]],
        split: str,
        dataset_dir: Union[str, None] = None,
        output_dir: Union[str, None] = None,
        no_cache: bool = True,
        debug: bool = False,
    ) -> None:
        """Initialize the command."""
        self.metrics = metrics
        self.providers = providers
        self.dataset = dataset
        self.split = split
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.no_cache = no_cache
        self.debug = debug

    def run(self) -> None:
        """Run the command."""
        try:
            if isinstance(self.providers, str) and "all" in self.providers.lower():
                self.providers = list(PROVIDERS.keys())
            else:
                for provider in self.providers:
                    if provider.lower() not in PROVIDERS.keys():
                        print(
                            error_message.format(
                                input_type="provider", user_input=provider
                            )
                        )
                        print(
                            "".join(
                                [f"  - [bold]{p}[bold]\n" for p in PROVIDERS.keys()]
                            )
                        )
                        exit(1)

            if self.dataset.lower() not in DATASETS.keys():
                print(
                    error_message.format(input_type="dataset", user_input=self.dataset)
                )
                print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS.keys()]))
                exit(1)

            for metric in self.metrics:
                if metric.upper() not in Metrics.__members__.keys():
                    print(error_message.format(input_type="metric", user_input=metric))
                    print(
                        "".join(
                            [
                                f"  - [bold]{m}[bold]\n"
                                for m in Metrics.__members__.keys()
                            ]
                        )
                    )
                    exit(1)

            _providers = [p.lower() for p in self.providers]
            _dataset = self.dataset.lower()
            _metrics = [m.upper() for m in self.metrics]

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

            print(f"🚧 Evaluation module not implemented yet. {_providers}, {_metrics}")
            exit(1)

        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)

        except Exception as e:
            raise Exception(e) from e
