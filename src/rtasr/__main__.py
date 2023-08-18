"""Define the entry point for the command line interface of rtasr."""

import argparse

from rich import print
from rich.traceback import install
from rich_argparse import RichHelpFormatter

from rtasr.cli import (
    BenchmarkASRCommand,
    DownloadDatasetCommand,
    ListItemsCommand,
)
from rtasr.cli_messages import ascii_art

install(show_locals=True)


def main() -> None:
    """Define the entry point for the command line interface of rtasr."""
    parser = argparse.ArgumentParser(
        prog="rtasr",
        description="🏆 Run benchmarks against the most common ASR tools on the market.",
        usage="rtasr <command> [<args>]",
        formatter_class=RichHelpFormatter,
    )
    commands_parser = parser.add_subparsers(dest="command")

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
    exit(0)


if __name__ == "__main__":
    main()
