"""Create plots for RTASR evaluation results."""

import datetime
import json
from collections import Counter
from enum import Enum
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


class ProviderNameDisplay(str, Enum):
    """The provider name display."""

    assemblyai = "AssemblyAI"
    aws = "AWS"
    azure = "Azure"
    deepgram = "Deepgram"
    google = "Google"
    revai = "Rev.ai"
    speechmatics = "Speechmatics"
    wordcab = "Wordcab"
    wordcab_hosted = "Wordcab Self-Hosted"


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


def plot_data_into_table(
    data: List[DataPoint], metric: str, dataset: str, output_dir: Path
) -> Path:
    """Plot the data into a table."""
    _data = [entry.model_dump() for entry in data]

    df = pd.DataFrame(_data)
    df = df.rename(
        columns={"asr_provider": "ASR Provider", "metric": "Metric", "value": "Value"}
    )

    df = df[df["Metric"] != "hits"]
    df = df[df["Metric"] != "false_alarms"]

    fig = plt.gcf()
    ax = plt.gca()

    all_providers = get_provider_names(data)
    all_metrics = get_metric_names(data)

    # Remove false_alarm and hits from the list of metrics to display
    if "false_alarm" in all_metrics:
        all_metrics.remove("false_alarm")
    if "hits" in all_metrics:
        all_metrics.remove("hits")

    columns = ["ASR Provider"] + all_metrics
    table_data = {provider: [] for provider in all_providers}
    for m in all_metrics:
        for provider in all_providers:
            all_values = df.loc[
                (df["ASR Provider"] == provider) & (df["Metric"] == m), "Value"
            ].values
            if len(all_values) > 0:
                table_data[provider].append(round(all_values.mean(), 3))
            else:
                table_data[provider].append(0)

    # Update provider names with display names
    _table_data = table_data.copy()
    for k in table_data.keys():
        k = k.replace("-", "_")
        _table_data[ProviderNameDisplay[k].value] = table_data.get(k)

    rows_data = [[provider] + _table_data[provider] for provider in all_providers]
    table = ax.table(
        cellText=rows_data, cellLoc="center", colLabels=columns, loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Change background color for header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#56b5fd")
        table[(0, i)].set_text_props(color="w")
        table[(0, i)].set_fontsize(12)

    # Make the best metric bold
    for i in range(1, len(all_metrics) + 1):
        if metric == "WER" or metric == "DER":
            best_metric = min([row[i] for row in rows_data])
        else:
            best_metric = max([row[i] for row in rows_data])

        for j in range(1, len(all_providers) + 1):
            if rows_data[j - 1][i] == best_metric:
                table[(j, i)].set_text_props(weight="bold")
            else:
                table[(j, i)].set_text_props(color="#444444")

    table.auto_set_column_width(col=list(range(len(columns))))

    fig.tight_layout()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Save the plot
    save_path = output_dir / f"{metric.lower()}_evaluation_{dataset.lower()}.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    return save_path
