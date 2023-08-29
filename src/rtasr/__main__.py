"""Define the entry point for the command line interface of rtasr."""

import argparse

from rich import print
from rich.traceback import install
from rich_argparse import RichHelpFormatter

from rtasr.cli import (
    AudioLengthCommand,
    DownloadDatasetCommand,
    EvaluationCommand,
    ListItemsCommand,
    PlotCommand,
    TranscriptionASRCommand,
)
from rtasr.cli_messages import ascii_art

install(show_locals=True)


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments passed to the command line interface."""
    parser = argparse.ArgumentParser(
        prog="rtasr",
        description="🏆 Run benchmarks against the most common ASR tools on the market.",
        usage="rtasr <command> [<args>]",
        formatter_class=RichHelpFormatter,
    )
    commands_parser = parser.add_subparsers(dest="command")

    # Register subcommands
    AudioLengthCommand.register_subcommand(commands_parser)
    DownloadDatasetCommand.register_subcommand(commands_parser)
    EvaluationCommand.register_subcommand(commands_parser)
    ListItemsCommand.register_subcommand(commands_parser)
    PlotCommand.register_subcommand(commands_parser)
    TranscriptionASRCommand.register_subcommand(commands_parser)

    return parser.parse_args()


def execute_command(args: argparse.Namespace) -> None:
    """Execute the command passed to the command line interface."""
    if not hasattr(args, "func"):
        print(
            "[bold red]Oops something went wrong. Please check the command you"
            f" entered.[/bold red]\n👉 {args}\n"
        )
        return 1

    args.func(args).run()
    return 0


def main() -> None:
    """CLI main wrapper."""
    args = parse_arguments()
    print(ascii_art)
    execute_command(args)


if __name__ == "__main__":
    main()  # pragma: no cover
