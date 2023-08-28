"""The list command."""

import argparse
from typing import Union

from rich import print
from typing_extensions import Literal

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, PROVIDERS, Metrics


def list_items_command_factory(args: argparse.Namespace):
    return ListItemsCommand(item_type=args.type)


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

    def __init__(
        self, item_type: Union[Literal["datasets", "metrics", "providers"], None] = None
    ) -> None:
        """Initialize the command."""
        self.item_type = item_type

    def run(self) -> None:
        """Run the command."""
        if self.item_type is None:
            self._print_all()
            exit(0)

        if self.item_type.lower() == "datasets":
            self._print_datasets()
        elif self.item_type.lower() == "metrics":
            self._print_metrics()
        elif self.item_type.lower() == "providers":
            self._print_providers()
        else:
            print(
                error_message.format(input_type="item type", user_input=self.item_type)
            )
            print("`datasets`, `metrics` or `providers`")
            exit(1)

    def _print_all(self) -> None:
        """Print all the items."""
        self._print_datasets()
        self._print_metrics()
        self._print_providers()

    def _print_datasets(self) -> None:
        """Print the datasets."""
        print("Datasets: splits (number of files):")
        for dataset in DATASETS.keys():
            splits_and_files = [
                f"{split} ({nb_files})"
                for split, nb_files in DATASETS[dataset]["number_of_files"].items()
            ]
            print(
                f"  - [bold]{dataset}[/bold]: {', '.join(splits_and_files)}\n",
                f"  compatible metrics -> {DATASETS[dataset]['metrics']}",
            )

    def _print_providers(self) -> None:
        """Print the providers."""
        print(
            "Providers: "
            + ", ".join([f"[bold magenta]{p}[/bold magenta]" for p in PROVIDERS.keys()])
        )

    def _print_metrics(self) -> None:
        """Print the metrics."""
        print(
            "Metrics: "
            + ", ".join(
                [f"[bold yellow]{m}[/bold yellow]" for m in Metrics.__members__.keys()]
            )
        )
