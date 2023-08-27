"""Test constants.py module."""

from typing import OrderedDict

import pytest

from rtasr.constants import DATASETS, PROVIDERS, Metrics


class TestConstants:
    """Test the DATASETS, PROVIDERS, and Metrics constants."""

    def test_datasets(self) -> None:
        """Test DATASETS constant."""
        assert isinstance(DATASETS, OrderedDict)

        assert "ami" in DATASETS
        assert "voxconverse" in DATASETS

    def test_datasets_ami(self) -> None:
        """Test AMI dataset."""
        assert DATASETS["ami"]["splits"] == ["train", "dev", "test"]
        assert DATASETS["ami"]["audio_types"] == ["Mix-Headset", "Array1-01"]
        assert DATASETS["ami"]["concurrency_limit"] == 5
        assert DATASETS["ami"]["speaker_map"] == "AMISpeakerMap"
        assert DATASETS["ami"]["exclude_ids"] == ["IS1007d", "IS1003b"]
        assert DATASETS["ami"]["urls"] == {
            "rttm": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm",
            "uem": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem",
            "list": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt",
        }
        assert DATASETS["ami"]["manifest_filepaths"] == {
            "test": [
                "test/manifest_Array1-01.json",
                "test/manifest_Mix-Headset.json",
            ],
            "dev": [
                "dev/manifest_Array1-01.json",
                "dev/manifest_Mix-Headset.json",
            ],
            "train": [
                "train/manifest_Array1-01.json",
                "train/manifest_Mix-Headset.json",
            ],
        }
        assert DATASETS["ami"]["rttm_filepaths"] == {
            "dev": "dev/rttm",
            "test": "test/rttm",
            "train": "train/rttm",
        }

    def test_datasets_voxconverse(self) -> None:
        """Test VoxConverse dataset."""
        assert DATASETS["voxconverse"]["splits"] == ["dev", "test"]
        assert DATASETS["voxconverse"]["speaker_map"] == "VoxConverseSpeakerMap"
        assert DATASETS["voxconverse"]["zip_urls"] == {
            "rttm": (
                "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip"
            ),
            "dev": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
            "test": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip",
        }
        assert DATASETS["voxconverse"]["filepaths"] == {
            "dev": "dev/audio",
            "test": "test/voxconverse_test_wav",
            "rttm": "rttm/voxconverse-master",
        }
        assert DATASETS["voxconverse"]["manifest_filepaths"] == {
            "dev": ["dev_manifest.json"],
            "test": ["test_manifest.json"],
        }
        assert DATASETS["voxconverse"]["rttm_filepaths"] == {
            "dev": "rttm/voxconverse-master/dev",
            "test": "rttm/voxconverse-master/test",
        }

    def test_providers(self) -> None:
        """Test PROVIDERS constant."""
        assert isinstance(PROVIDERS, OrderedDict)

        assert "assemblyai" in PROVIDERS
        assert "aws" in PROVIDERS
        assert "azure" in PROVIDERS
        assert "deepgram" in PROVIDERS
        assert "google" in PROVIDERS
        assert "revai" in PROVIDERS
        assert "speechmatics" in PROVIDERS
        assert "wordcab" in PROVIDERS

    def test_providers_assemblyai(self) -> None:
        """Test AssemblyAI provider."""
        assert PROVIDERS["assemblyai"]["url"] == "https://api.assemblyai.com/v2"
        assert PROVIDERS["assemblyai"]["engine"] == "AssemblyAI"
        assert PROVIDERS["assemblyai"]["output"] == "AssemblyAIOutput"
        assert PROVIDERS["assemblyai"]["speaker_map"] == "AssemblyAISpeakerMap"
        assert PROVIDERS["assemblyai"]["concurrency_limit"] == 5
        assert PROVIDERS["assemblyai"]["options"] == {
            "speaker_labels": True,
            "punctuate": True,
        }

    def test_providers_aws(self) -> None:
        """Test AWS provider."""
        assert PROVIDERS["aws"]["url"] == ""
        assert PROVIDERS["aws"]["engine"] == "Aws"
        assert PROVIDERS["aws"]["output"] == "AwsOutput"
        assert PROVIDERS["aws"]["speaker_map"] == "AwsSpeakerMap"
        assert PROVIDERS["aws"]["options"] == {}

    def test_providers_azure(self) -> None:
        """Test Azure provider."""
        assert PROVIDERS["azure"]["url"] == ""
        assert PROVIDERS["azure"]["engine"] == "Azure"
        assert PROVIDERS["azure"]["output"] == "AzureOutput"
        assert PROVIDERS["azure"]["speaker_map"] == "AzureSpeakerMap"
        assert PROVIDERS["azure"]["options"] == {}

    def test_providers_deepgram(self) -> None:
        """Test Deepgram provider."""
        assert PROVIDERS["deepgram"]["url"] == "https://api.deepgram.com/v1/listen"
        assert PROVIDERS["deepgram"]["engine"] == "Deepgram"
        assert PROVIDERS["deepgram"]["output"] == "DeepgramOutput"
        assert PROVIDERS["deepgram"]["speaker_map"] == "DeepgramSpeakerMap"
        assert PROVIDERS["deepgram"]["concurrency_limit"] == 5
        assert PROVIDERS["deepgram"]["options"] == {
            "diarize": True,
            "model": "nova",
            "punctuate": True,
            "utterances": True,
        }

    def test_providers_google(self) -> None:
        """Test Google provider."""
        assert PROVIDERS["google"]["url"] == ""
        assert PROVIDERS["google"]["engine"] == "Google"
        assert PROVIDERS["google"]["output"] == "GoogleOutput"
        assert PROVIDERS["google"]["speaker_map"] == "GoogleSpeakerMap"
        assert PROVIDERS["google"]["options"] == {}

    def test_providers_revai(self) -> None:
        """Test Rev.ai provider."""
        assert PROVIDERS["revai"]["url"] == "https://api.rev.ai/speechtotext/v1"
        assert PROVIDERS["revai"]["engine"] == "RevAI"
        assert PROVIDERS["revai"]["output"] == "RevAIOutput"
        assert PROVIDERS["revai"]["speaker_map"] == "RevAISpeakerMap"
        assert PROVIDERS["revai"]["concurrency_limit"] == 5
        assert PROVIDERS["revai"]["options"] == {
            "remove_disfluencies": False,
            "skip_diarization": False,
            "skip_postprocessing": False,
            "skip_punctuation": False,
            "transcriber": "machine",
        }

    def test_providers_speechmatics(self) -> None:
        """Test Speechmatics provider."""
        assert PROVIDERS["speechmatics"]["url"] == "https://asr.api.speechmatics.com/v2"
        assert PROVIDERS["speechmatics"]["engine"] == "Speechmatics"
        assert PROVIDERS["speechmatics"]["output"] == "SpeechmaticsOutput"
        assert PROVIDERS["speechmatics"]["speaker_map"] == "SpeechmaticsSpeakerMap"
        assert PROVIDERS["speechmatics"]["concurrency_limit"] == 1
        assert PROVIDERS["speechmatics"]["options"] == {
            "type": "transcription",
            "transcription_config": {
                "diarization": "speaker",
                "language": "en",
                "operating_point": "enhanced",
            },
        }

    def test_providers_wordcab(self) -> None:
        """Test Wordcab provider."""
        assert PROVIDERS["wordcab"]["url"] == "https://wordcab.com/api/v1"
        assert PROVIDERS["wordcab"]["engine"] == "Wordcab"
        assert PROVIDERS["wordcab"]["output"] == "WordcabOutput"
        assert PROVIDERS["wordcab"]["speaker_map"] == "WordcabSpeakerMap"
        assert PROVIDERS["wordcab"]["concurrency_limit"] == 10
        assert PROVIDERS["wordcab"]["options"] == {
            "alignment": False,
            "diarize": True,
            "dual_channel": False,
            "only_api": False,
            "source": "audio",
        }

    def test_metrics(self) -> None:
        """Test Metrics enum."""
        assert Metrics.__members__ == {
            "DER": "DER",
            "WER": "WER",
        }

        assert Metrics.DER == "DER"
        assert Metrics.WER == "WER"

        assert Metrics.DER == Metrics("DER")
        assert Metrics.WER == Metrics("WER")

        with pytest.raises(ValueError):
            Metrics("der")
        with pytest.raises(ValueError):
            Metrics("wer")
