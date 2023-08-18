"""Define all the constants used in rtasr."""

from typing import OrderedDict

DATASETS = ["ami", "voxconverse"]
PROVIDERS = OrderedDict(
    [
        ("assemblyai", ""),
        ("aws", ""),
        ("azure", ""),
        ("deepgram", "https://api.deepgram.com/v1/listen"),
        ("google", ""),
        ("revai", ""),
        ("speechmatics", ""),
        ("wordcab", ""),
    ]
)
