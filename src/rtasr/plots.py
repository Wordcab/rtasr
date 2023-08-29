"""Create plots for RTASR evaluation results."""

import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel

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

        if eval_file_paths.exists():
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


def get_provider_names(data: List[DataPoint]) -> List[str]:
    """Get a list of provider names."""
    provider_names = []
    for entry in data:
        if entry.asr_provider not in provider_names:
            provider_names.append(entry.asr_provider)

    return provider_names


def count_files_per_provider(data: List[DataPoint], metric: str) -> Dict[str, int]:
    """Count the number of files per provider."""
    counter = Counter([entry.asr_provider for entry in data if entry.metric == metric])

    return counter


def plot_data_point_distribution(
    data: List[DataPoint], metric: str, dataset: str, output_dir: Path
) -> Path:
    """Plot the distribution of data points."""
    _data = [entry.model_dump() for entry in data]

    df = pd.DataFrame(_data)
    df = df.rename(
        columns={"asr_provider": "ASR Provider", "metric": "Metric", "value": "Value"}
    )

    df = df[df["Metric"] != "hits"]
    df = df[df["Metric"] != "false_alarms"]

    metric_names = get_metric_names(data)

    if metric == "DER":
        metric_names.remove("der")
        metric_names.remove("false_alarm")
        metric_names.insert(0, "der")
    elif metric == "WER":
        metric_names.remove("wer")
        metric_names.remove("hits")
        metric_names.insert(0, "wer")

    files_per_provider = count_files_per_provider(data, metric.lower())

    # Rename the providers by the name with (n) files
    for provider in files_per_provider.keys():
        df.loc[
            df["ASR Provider"] == provider, "ASR Provider"
        ] = f"{provider} ({files_per_provider[provider]})"

    providers = get_provider_names(data)
    providers = [
        f"{provider} ({files_per_provider[provider]})" for provider in providers
    ]

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Create a grouped bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df,
        x="ASR Provider",
        y="Value",
        hue="Metric",
        order=providers,
        hue_order=metric_names,
        palette="spring",
    )
    plt.title(
        f"{metric} Evaluation Metrics for {dataset.upper()} Dataset -"
        f" {datetime.date.today()}"
    )
    plt.ylabel("Value normalized between 0 and 1 (lower is better)")
    plt.xlabel("ASR Provider (number of files evaluated)")
    plt.tight_layout()

    # Save the plot
    save_path = output_dir / f"{metric.lower()}_evaluation_{dataset.lower()}.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    return save_path
