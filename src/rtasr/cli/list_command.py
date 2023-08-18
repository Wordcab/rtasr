"""The list command."""

import argparse
from typing import Union

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, PROVIDERS


def list_items_command_factory(args: argparse.Namespace):
    return ListItemsCommand(args.type)


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
            print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS.keys()]))
            exit(1)
        if self.item_type.lower() == "datasets":
            print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS]))
        elif self.item_type.lower() == "providers":
            print("".join([f"  - [bold]{p}[bold]\n" for p in PROVIDERS.keys()]))
        else:
            print(
                error_message.format(input_type="item type", user_input=self.item_type)
            )
            print("`datasets` or `providers`")
            exit(1)
