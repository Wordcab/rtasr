"""Define the entry point for the command line interface of rtasr."""

import argparse

from rich.console import Console


ascii_art = r"""
  ___      _         _____ _         _       _   ___ ___ 
 | _ \__ _| |_ ___  |_   _| |_  __ _| |_    /_\ / __| _ \
 |   / _` |  _/ -_)   | | | ' \/ _` |  _|  / _ \\__ |   /
 |_|_\__,_|\__\___|   |_| |_||_\__,_|\__| /_/ \_|___|_|_\
 _________________________________________________________
 by Wordcab

"""
                                                      
console = Console()


def main() -> None:
    """Define the entry point for the command line interface of rtasr."""
    parser = argparse.ArgumentParser("Rate That ASR", usage="rtasr <command> [<args>]")
    commands_parser = parser.add_subparsers(help="rtasr command helpers")

    # Register subcommands

    args = parser.parse_args()

    console.print(ascii_art, justify="left", style="bold blue")

    if not hasattr(args, "func"):
        console.print(
            "[bold red]Error: Something went wrong. Please check the command you entered.[/bold red]\n"
            "[bold red]==================================================================[/bold red]\n"
        )
        console.print(f"How to use the CLI 👇\n\n{parser.format_help()}")
        exit(1)

    args.func(args).run()


if __name__ == "__main__":
    main()
