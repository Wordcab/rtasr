"""Define all the constants used in rtasr."""

from enum import Enum
from typing import OrderedDict

DATASETS = OrderedDict(
    [
        (
            "ami",
            {
                "splits": ["train", "dev", "test"],
                "concurrency_limit": 1,
                "speaker_map": "AMISpeakerMap",
                "urls": {
                    "rttm": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm",
                    "uem": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem",
                    "list": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt",
                },
                "exclude_ids": ["IS1007d", "IS1003b"],
                "manifest_filepaths": {
                    "test": "test/manifest.json",
                    "dev": "dev/manifest.json",
                    "train": "train/manifest.json",
                },
                "rttm_filepaths": {
                    "dev": "dev/rttm",
                    "test": "test/rttm",
                    "train": "train/rttm",
                },
                "metrics": ["der", "wer"],
                "number_of_files": {
                    "train": 134,
                    "dev": 18,
                    "test": 16,
                },
            },
        ),
        (
            "fleurs",
            {
                "splits": ["train", "validation", "test"],
                "metrics": ["wer"],
                "number_of_files": {
                    "train": 2602,
                    "validation": 394,
                    "test": 647,
                },
            },
        ),
        (
            "voxconverse",
            {
                "splits": ["dev", "test"],
                "speaker_map": "VoxConverseSpeakerMap",
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
                "manifest_filepaths": {
                    "dev": "dev_manifest.json",
                    "test": "test_manifest.json",
                },
                "rttm_filepaths": {
                    "dev": "rttm/voxconverse-master/dev",
                    "test": "rttm/voxconverse-master/test",
                },
                "metrics": ["der"],
                "number_of_files": {
                    "dev": 216,
                    "test": 232,
                },
            },
        ),
    ]
)
PROVIDERS = OrderedDict(
    [
        (
            "assemblyai",
            {
                "url": "https://api.assemblyai.com/v2",
                "engine": "AssemblyAI",
                "output": "AssemblyAIOutput",
                "speaker_map": "AssemblyAISpeakerMap",
                "concurrency_limit": 5,
                "options": {
                    "speaker_labels": True,
                    "punctuate": True,
                },
            },
        ),
        (
            "aws",
            {
                "url": "",
                "engine": "Aws",
                "output": "AwsOutput",
                "speaker_map": "AwsSpeakerMap",
                "options": {},
            },
        ),
        (
            "azure",
            {
                "url": "",
                "engine": "Azure",
                "output": "AzureOutput",
                "speaker_map": "AzureSpeakerMap",
                "options": {},
            },
        ),
        (
            "deepgram",
            {
                "url": "https://api.deepgram.com/v1/listen",
                "engine": "Deepgram",
                "output": "DeepgramOutput",
                "speaker_map": "DeepgramSpeakerMap",
                "concurrency_limit": 5,
                "options": {
                    "diarize": True,
                    "model": "nova",
                    "punctuate": True,
                    "utterances": True,
                },
            },
        ),
        (
            "google",
            {
                "url": "",
                "engine": "Google",
                "output": "GoogleOutput",
                "speaker_map": "GoogleSpeakerMap",
                "options": {},
            },
        ),
        (
            "revai",
            {
                "url": "https://api.rev.ai/speechtotext/v1",
                "engine": "RevAI",
                "output": "RevAIOutput",
                "speaker_map": "RevAISpeakerMap",
                "concurrency_limit": 5,
                "options": {
                    "remove_disfluencies": False,
                    "skip_diarization": False,
                    "skip_postprocessing": False,
                    "skip_punctuation": False,
                    "transcriber": "machine",
                },
            },
        ),
        (
            "speechmatics",
            {
                "url": "https://asr.api.speechmatics.com/v2",
                "engine": "Speechmatics",
                "output": "SpeechmaticsOutput",
                "speaker_map": "SpeechmaticsSpeakerMap",
                "concurrency_limit": 5,
                "options": {
                    "type": "transcription",
                    "transcription_config": {
                        "diarization": "speaker",
                        "language": "en",
                        "operating_point": "enhanced",
                    },
                },
            },
        ),
        (
            "wordcab",
            {
                "url": "https://wordcab.com/api/v1",
                "engine": "Wordcab",
                "output": "WordcabOutput",
                "speaker_map": "WordcabSpeakerMap",
                "concurrency_limit": 5,
                "options": {
                    "alignment": False,
                    "diarize": True,
                    "dual_channel": False,
                    "only_api": False,
                    "source": "audio",
                },
            },
        ),
    ]
)


class Metrics(str, Enum):
    """Define the metrics."""

    DER = "DER"
    WER = "WER"
