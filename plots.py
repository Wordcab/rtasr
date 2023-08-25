import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = []
asr_providers = ["assemblyai", "deepgram", "revai", "speechmatics", "wordcab"]
for provider in asr_providers:
    _pa = Path.home() / ".cache" / "rtasr" / "evaluation" / "voxconverse" / "test" / provider
    for p in _pa.iterdir():
        if p.is_file() and p.suffix == ".json":
            with open(p, "r") as f:
                _data = json.load(f)
                _data.pop("false_alarm")
                filename = p.name.split(".")[0]
                data.append({"asr_provider": provider, "file": filename, "metrics": _data})

# Extract metric names and values
metric_names = list(data[0]["metrics"].keys())

# Create a list to store data for seaborn
seaborn_data = []

for provider in asr_providers:
    for entry in data:
        if entry["asr_provider"] == provider:
            for metric_name, metric_value in entry["metrics"].items():
                seaborn_data.append({
                    "ASR Provider": provider,
                    "Metric": metric_name,
                    "Value": metric_value
                })

# Create a DataFrame for Seaborn
df = pd.DataFrame(seaborn_data)

# Reorder metric_names to prioritize "der" metric
metric_names.remove("der")
metric_names.insert(0, "der")

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="ASR Provider", y="Value", hue="Metric", order=asr_providers, hue_order=metric_names, palette="spring")
plt.title("ASR Evaluation Metrics")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
