"""Define all the constants used in rtasr."""

from typing import OrderedDict

DATASETS = OrderedDict(
    [
        (
            "ami",
            {
                "splits": ["test", "dev", "train"],
                "audio_types": ["Mix-Headset", "Array1-01"],
                "urls": {
                    "rttm": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm",
                    "uem": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem",
                    "list": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt",
                },
                "exclude_ids": ["IS1007d", "IS1003b"],
            },
        ),
        (
            "voxconverse",
            {
                "splits": ["dev", "test"],
                "zip_urls": {
                    "rttm": "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip",
                    "dev": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
                    "test": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip",
                },
                "filepaths": {
                    "dev": "dev/audio",
                    "test": "test/voxconverse_test_wav",
                    "rttm": "rttm/voxconverse-master",
                },
            },
        ),
    ]
)
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
