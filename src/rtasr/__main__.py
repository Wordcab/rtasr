"""Define the entry point for the command line interface of rtasr."""

import argparse
import asyncio
from typing import Union

from rich import print
from rich.traceback import install
from rich_argparse import RichHelpFormatter

install(show_locals=True)


ascii_art = r"""
  ___      _         _____ _         _       _   ___ ___
 | _ \__ _| |_ ___  |_   _| |_  __ _| |_    /_\ / __| _ \
 |   / _` |  _/ -_)   | | | ' \/ _` |  _|  / _ \\__ |   /
 |_|_\__,_|\__\___|   |_| |_||_\__,_|\__| /_/ \_|___|_|_\
 _________________________________________________________
 by Wordcab
"""


def download_dataset_command_factory(args: argparse.Namespace):
    return DownloadDatasetCommand(args.dataset, args.output_dir, args.no_cache)


class DownloadDatasetCommand:
    """Download a dataset."""

    DATASETS = ["ami", "voxconverse"]

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
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
                print(
                    f"[bold red]Error: The dataset `{self.dataset}` is not"
                    " supported.[/bold red]\n[bold"
                    " red]==================================================================[/bold"
                    " red]\n"
                )
                print("Do you mean one of these datasets?\n")
                print("".join([f"  - [bold]{d}[bold]\n" for d in self.DATASETS]))
                exit(1)
        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)
        except Exception as e:
            raise Exception(e) from e


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
    DownloadDatasetCommand.register_subcommand(commands_parser)

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
