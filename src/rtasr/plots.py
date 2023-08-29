"""Create plots for RTASR evaluation results."""

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from rich import print

from rtasr.constants import PROVIDERS


class DataPoint(BaseModel):
    """A data point."""

    asr_provider: str
    metric: str
    value: float


def load_data_from_cache(
    eval_metric: str,
    dataset: str,
    split: str,
    evaluation_dir: Path,
) -> List[DataPoint]:
    """Load data from cache."""
    data = []
    for provider in PROVIDERS.keys():
        eval_file_paths = evaluation_dir / dataset / split / provider / eval_metric

        for evaluation in eval_file_paths.iterdir():
            if evaluation.is_file() and evaluation.suffix == ".json":
                with open(evaluation, "r") as f:
                    _data = json.load(f)

                    for metric in _data.keys():
                        data.append(
                            DataPoint(
                                asr_provider=provider,
                                metric=metric,
                                value=_data[metric],
                            )
                        )

    return data


def get_metric_names(data: List[DataPoint]) -> List[str]:
    """Get a list of metric names."""
    metric_names = []
    for entry in data:
        if entry.metric not in metric_names:
            metric_names.append(entry.metric)

    return metric_names


def plot_data_point_distribution(data: List[DataPoint]) -> None:
    """Plot the distribution of data points."""
    # Create a DataFrame for Seaborn
    df = pd.DataFrame(data)

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Create a grouped bar plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Value", hue="Metric", palette="spring")
    plt.title("ASR Evaluation Metrics")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()




# df = pd.DataFrame(seaborn_data)

# # Reorder metric_names to prioritize "der" metric
# metric_names.remove("der")
# metric_names.insert(0, "der")

# # Set Seaborn style
# sns.set_theme(style="whitegrid")

# # Create a grouped bar plot
# plt.figure(figsize=(10, 6))
# sns.barplot(data=df, x="ASR Provider", y="Value", hue="Metric", order=asr_providers, hue_order=metric_names, palette="spring")
# plt.title("ASR Evaluation Metrics")
# plt.ylabel("Value")
# plt.tight_layout()
# plt.show()
