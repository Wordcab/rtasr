"""The benchmark command."""

import argparse
from typing import Union

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, PROVIDERS


def benchmark_asr_command_factory(args: argparse.Namespace):
    return BenchmarkASRCommand(
        args.provider, args.dataset, args.output_dir, args.no_cache
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
            "-o",
            "--output_dir",
            help=(
                "Path to store the downloaded files. Defaults to"
                " `~/.cache/rtasr/datasets`."
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
        output_dir: Union[str, None] = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the command."""
        self.provider = provider
        self.dataset = dataset
        self.output_dir = output_dir
        self.use_cache = use_cache

    def run(self) -> None:
        """Run the command."""
        try:
            if self.provider.lower() not in PROVIDERS:
                print(
                    error_message.format(
                        input_type="provider", user_input=self.provider
                    )
                )
                print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS]))
                exit(1)
            if self.dataset.lower() not in DATASETS:
                print(
                    error_message.format(input_type="dataset", user_input=self.dataset)
                )
                print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS]))
                exit(1)

            if self.provider.lower() == "deepgram":
                pass
            elif self.provider.lower() == "wordcab":
                pass
            else:
                raise NotImplementedError(
                    "The provider must be either `deepgram` or `wordcab`."
                )
        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)
        except Exception as e:
            raise Exception(e) from e
