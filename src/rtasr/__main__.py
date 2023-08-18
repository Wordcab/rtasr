"""Define the entry point for the command line interface of rtasr."""

import argparse
import asyncio
from typing import Union

from rich import print
from rich.traceback import install
from rich_argparse import RichHelpFormatter

install(show_locals=True)

from rtasr.cli_messages import ascii_art, error_message
from rtasr.constants import DATASETS, PROVIDERS


def benchmark_asr_command_factory(args: argparse.Namespace):
    return BenchmarkASRCommand(args.provider, args.dataset, args.output_dir, args.no_cache)


def download_dataset_command_factory(args: argparse.Namespace):
    return DownloadDatasetCommand(args.dataset, args.output_dir, args.no_cache)


def list_items_command_factory(args: argparse.Namespace):
    return ListItemsCommand(args.type)


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
                print(error_message.format(input_type="provider", user_input=self.provider))
                print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS]))
                exit(1)
            if self.dataset.lower() not in DATASETS:
                print(error_message.format(input_type="dataset", user_input=self.dataset))
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

class DownloadDatasetCommand:
    """Download a dataset."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        subparser = parser.add_parser("download", help="Download a dataset.")
        subparser.add_argument(
            "-d", "--dataset", help="The dataset to download.", required=True, type=str
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
        subparser.set_defaults(func=download_dataset_command_factory)

    def __init__(
        self,
        dataset: str,
        output_dir: Union[str, None] = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the command."""
        self.dataset = dataset
        self.output_dir = output_dir
        self.use_cache = use_cache

    def run(self) -> None:
        """Run the command."""
        try:
            if self.dataset.lower() == "ami":
                from rtasr.datasets import prepare_ami_dataset

                asyncio.run(prepare_ami_dataset(self.output_dir, self.use_cache))

            elif self.dataset.lower() == "voxconverse":
                from rtasr.datasets import prepare_voxconverse_dataset

                asyncio.run(
                    prepare_voxconverse_dataset(self.output_dir, self.use_cache)
                )

            else:
                print(error_message.format(input_type="dataset", user_input=self.dataset))
                print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS]))
                exit(1)
        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)
        except Exception as e:
            raise Exception(e) from e

class ListItemsCommand:
    """List items."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        """Register the subcommand."""
        subparser = parser.add_parser("list", help="List items.")
        subparser.add_argument(
            "-t",
            "--type",
            help="The type of items to list.",
            type=str,
            default=None,
        )
        subparser.set_defaults(func=list_items_command_factory)

    def __init__(self, item_type: Union[str, None] = None) -> None:
        """Initialize the command."""
        self.item_type = item_type

    def run(self) -> None:
        """Run the command."""
        if self.item_type is None:
            print("Datasets:")
            print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS]))
            print("Providers:")
            print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS]))
            exit(1)
        if self.item_type.lower() == "datasets":
            print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS]))
        elif self.item_type.lower() == "providers":
            print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS]))
        else:
            print(error_message.format(input_type="item type", user_input=self.item_type))
            print("`datasets` or `providers`")
            exit(1)


def main() -> None:
    """Define the entry point for the command line interface of rtasr."""
    parser = argparse.ArgumentParser(
        prog="rtasr",
        description="🏆 Run benchmarks against the most common ASR tools on the market.",
        usage="rtasr <command> [<args>]",
        formatter_class=RichHelpFormatter,
    )
    commands_parser = parser.add_subparsers(help="rtasr command helpers")

    # Register subcommands
    BenchmarkASRCommand.register_subcommand(commands_parser)
    DownloadDatasetCommand.register_subcommand(commands_parser)
    ListItemsCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    print(ascii_art)

    if not hasattr(args, "func"):
        print(
            "[bold red]Oops something went wrong. Please check the command you"
            f" entered.[/bold red]\n👉 {args}\n"
        )
        parser.print_help()
        exit(1)

    args.func(args).run()


if __name__ == "__main__":
    main()
