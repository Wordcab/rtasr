"""The plot command."""

import argparse
from pathlib import Path
from typing import List, Union

from rich import print
from typing_extensions import Literal

from rtasr.cli_messages import error_message
from rtasr.constants import DATASETS, Metrics
from rtasr.plots import (
    DataPoint,
    load_data_from_cache,
    plot_data_into_table,
    plot_data_point_distribution,
)


def plot_command_factory(args: argparse.Namespace):
    return PlotCommand(
        metric=args.metric,
        plot_type=args.plot_type,
        dataset=args.dataset,
        split=args.split,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
    )


class PlotCommand:
    """Plot evaluation results."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        """Register the subcommand."""
        subparser = parser.add_parser("plot", help="Plot evaluation results.")
        subparser.add_argument(
            "-m",
            "--metric",
            help="The metric to plot.",
            type=str,
            required=True,
        )
        subparser.add_argument(
            "-t",
            "--plot_type",
            help="The plot type to use. Defaults to `graph`.",
            type=str,
            required=True,
            default="graph",
            choices=["graph", "table"],
        )
        subparser.add_argument(
            "-d",
            "--dataset",
            help="The dataset to plot.",
            type=str,
            required=True,
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
            "--evaluation_dir",
            help=(
                "Path where the evaluation files are stored. Defaults to"
                " `~/.cache/rtasr/evaluation`."
            ),
            required=False,
            default=None,
            type=str,
        )
        subparser.add_argument(
            "--output_dir",
            help=(
                "Path where the plots are stored. Defaults to the current working"
                " directory."
            ),
            default=None,
            type=str,
        )
        subparser.set_defaults(func=plot_command_factory)

    def __init__(
        self,
        metric: str,
        plot_type: Literal["graph", "table"],
        dataset: str,
        split: str,
        evaluation_dir: Union[str, None] = None,
        output_dir: Union[str, None] = None,
    ) -> None:
        """Initialize the command."""
        self.metric = metric
        self.plot_type = plot_type
        self.dataset = dataset
        self.split = split
        self.evaluation_dir = evaluation_dir
        self.output_dir = output_dir

    def run(self) -> None:
        """Run the command."""
        try:
            if self.dataset.lower() not in DATASETS.keys():
                print(
                    error_message.format(input_type="dataset", user_input=self.dataset)
                )
                print("".join([f"  - [bold]{d}[bold]\n" for d in DATASETS.keys()]))
                exit(1)

            _dataset = self.dataset.lower()

            if self.metric.upper() not in Metrics.__members__.keys():
                print(error_message.format(input_type="metric", user_input=self.metric))
                print(
                    "".join(
                        [f"  - [bold]{m}[bold]\n" for m in Metrics.__members__.keys()]
                    )
                )
                exit(1)

            _metric = Metrics[self.metric.upper()]

            if self.metric.lower() not in DATASETS[_dataset]["metrics"]:
                print(
                    f"[bold red]Metric {_metric} not supported for dataset {_dataset}."
                    " [/bold red]\nPlease check `rtasr list -t datasets` for more"
                    " info."
                )
                exit(1)

            if self.evaluation_dir is None:
                self.evaluation_dir = Path.home() / ".cache" / "rtasr" / "evaluation"

            if self.output_dir is None:
                self.output_dir = Path.cwd()

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

            data: List[DataPoint] = []
            for split in splits:
                data.extend(
                    load_data_from_cache(
                        eval_metric=_metric.value,
                        dataset=_dataset,
                        split=split,
                        evaluation_dir=self.evaluation_dir,
                    )
                )

            if len(data) == 0:
                print(
                    f"[bold red]No data found for metric {_metric} and dataset"
                    f" {_dataset}.[/bold red]"
                )
                exit(1)

            if self.plot_type == "table":
                save_path = plot_data_into_table(
                    data=data,
                    metric=_metric,
                    dataset=_dataset,
                    output_dir=self.output_dir,
                )
            else:
                save_path = plot_data_point_distribution(
                    data=data,
                    metric=_metric,
                    dataset=_dataset,
                    output_dir=self.output_dir,
                )

            print(f"Plot saved to [bold]{save_path}[/bold].")

        except KeyboardInterrupt:
            print("\n[bold red]Cancelled by user.[/bold red]\n")
            exit(1)

        except Exception as e:
            raise Exception(e) from e
