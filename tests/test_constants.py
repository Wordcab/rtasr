"""Test constants.py module."""

from typing import OrderedDict

import pytest

from rtasr.constants import DATASETS, PROVIDERS, Metrics


class TestConstants:
    """Test the DATASETS, PROVIDERS, and Metrics constants."""

    def test_datasets(self) -> None:
        """Test DATASETS constant."""
        assert isinstance(DATASETS, OrderedDict)
        assert len(DATASETS) == 3

        assert "ami" in DATASETS
        assert "fleurs" in DATASETS
        assert "voxconverse" in DATASETS

    def test_datasets_ami(self) -> None:
        """Test AMI dataset."""
        assert DATASETS["ami"]["splits"] == ["train", "dev", "test"]
        assert DATASETS["ami"]["concurrency_limit"] == 1
        assert DATASETS["ami"]["speaker_map"] == "AMISpeakerMap"
        assert DATASETS["ami"]["exclude_ids"] == ["IS1007d", "IS1003b"]
        assert DATASETS["ami"]["urls"] == {
            "rttm": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm",
            "uem": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem",
            "list": "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt",
        }
        assert DATASETS["ami"]["manifest_filepaths"] == {
            "test": "test/manifest.json",
            "dev": "dev/manifest.json",
            "train": "train/manifest.json",
        }
        assert DATASETS["ami"]["rttm_filepaths"] == {
            "dev": "dev/rttm",
            "test": "test/rttm",
            "train": "train/rttm",
        }
        assert DATASETS["ami"]["metrics"] == ["der", "wer", "wrr"]
        assert DATASETS["ami"]["number_of_files"] == {
            "train": 134,
            "dev": 18,
            "test": 16,
        }

    def test_datasets_fleurs(self) -> None:
        """Test Fleurs dataset."""
        assert DATASETS["fleurs"]["splits"] == ["train", "validation", "test"]
        assert DATASETS["fleurs"]["metrics"] == ["wer", "wrr"]
        assert DATASETS["fleurs"]["number_of_files"] == {
            "train": 2602,
            "validation": 394,
            "test": 647,
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
            "dev": "dev_manifest.json",
            "test": "test_manifest.json",
        }
        assert DATASETS["voxconverse"]["rttm_filepaths"] == {
            "dev": "rttm/voxconverse-master/dev",
            "test": "rttm/voxconverse-master/test",
        }
        assert DATASETS["voxconverse"]["metrics"] == ["der"]
        assert DATASETS["voxconverse"]["number_of_files"] == {
            "dev": 216,
            "test": 232,
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
        assert PROVIDERS["assemblyai"]["pricing"] == {
            "value": 0.000181,
            "unit": "second",
        }

    def test_providers_aws(self) -> None:
        """Test AWS provider."""
        assert PROVIDERS["aws"]["url"] == ""
        assert PROVIDERS["aws"]["engine"] == "Aws"
        assert PROVIDERS["aws"]["output"] == "AwsOutput"
        assert PROVIDERS["aws"]["speaker_map"] == "AwsSpeakerMap"
        assert PROVIDERS["aws"]["options"] == {}
        assert PROVIDERS["aws"]["pricing"] == {}

    def test_providers_azure(self) -> None:
        """Test Azure provider."""
        assert PROVIDERS["azure"]["url"] == ""
        assert PROVIDERS["azure"]["engine"] == "Azure"
        assert PROVIDERS["azure"]["output"] == "AzureOutput"
        assert PROVIDERS["azure"]["speaker_map"] == "AzureSpeakerMap"
        assert PROVIDERS["azure"]["options"] == {}
        assert PROVIDERS["azure"]["pricing"] == {}

    def test_providers_deepgram(self) -> None:
        """Test Deepgram provider."""
        assert PROVIDERS["deepgram"]["url"] == "https://api.deepgram.com/v1/listen"
        assert PROVIDERS["deepgram"]["engine"] == "Deepgram"
        assert PROVIDERS["deepgram"]["output"] == "DeepgramOutput"
        assert PROVIDERS["deepgram"]["speaker_map"] == "DeepgramSpeakerMap"
        assert PROVIDERS["deepgram"]["concurrency_limit"] == 5
        assert PROVIDERS["deepgram"]["options"] == {
            "diarize": True,
            "model": "nova-2-ea",
            "punctuate": True,
            "utterances": True,
        }
        assert PROVIDERS["deepgram"]["pricing"] == {
            "value": 0.0044,
            "unit": "minute",
        }

    def test_providers_elevateai(self) -> None:
        """Test ElevateAI provider."""
        assert PROVIDERS["elevateai"]["url"] == "https://api.elevateai.com/v1/interactions"
        assert PROVIDERS["elevateai"]["engine"] == "ElevateAI"
        assert PROVIDERS["elevateai"]["output"] == "ElevateAIOutput"
        assert PROVIDERS["elevateai"]["speaker_map"] == "ElevateAISpeakerMap"
        assert PROVIDERS["elevateai"]["concurrency_limit"] == 5
        assert PROVIDERS["elevateai"]["options"] == {
            "type": "audio",
            "languageTag": "en",
            "vertical": "default",
            "audioTranscriptionMode": "highAccuracy",
            "includeAiResults": True,
        }
        assert PROVIDERS["elevateai"]["pricing"] == {
            "value": 0.0030,
            "unit": "minute",
        }

    def test_providers_google(self) -> None:
        """Test Google provider."""
        assert PROVIDERS["google"]["url"] == ""
        assert PROVIDERS["google"]["engine"] == "Google"
        assert PROVIDERS["google"]["output"] == "GoogleOutput"
        assert PROVIDERS["google"]["speaker_map"] == "GoogleSpeakerMap"
        assert PROVIDERS["google"]["options"] == {}
        assert PROVIDERS["google"]["pricing"] == {}

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
        assert PROVIDERS["revai"]["pricing"] == {
            "value": 0.02,
            "unit": "minute",
        }

    def test_providers_speechmatics(self) -> None:
        """Test Speechmatics provider."""
        assert PROVIDERS["speechmatics"]["url"] == "https://asr.api.speechmatics.com/v2"
        assert PROVIDERS["speechmatics"]["engine"] == "Speechmatics"
        assert PROVIDERS["speechmatics"]["output"] == "SpeechmaticsOutput"
        assert PROVIDERS["speechmatics"]["speaker_map"] == "SpeechmaticsSpeakerMap"
        assert PROVIDERS["speechmatics"]["concurrency_limit"] == 5
        assert PROVIDERS["speechmatics"]["options"] == {
            "type": "transcription",
            "transcription_config": {
                "diarization": "speaker",
                "language": "en",
                "operating_point": "enhanced",
            },
        }
        assert PROVIDERS["speechmatics"]["pricing"] == {
            "value": 0.0174,
            "unit": "minute",
        }

    def test_providers_wordcab(self) -> None:
        """Test Wordcab provider."""
        assert PROVIDERS["wordcab"]["url"] == "https://wordcab.com/api/v1"
        assert PROVIDERS["wordcab"]["engine"] == "Wordcab"
        assert PROVIDERS["wordcab"]["output"] == "WordcabOutput"
        assert PROVIDERS["wordcab"]["speaker_map"] == "WordcabSpeakerMap"
        assert PROVIDERS["wordcab"]["concurrency_limit"] == 5
        assert PROVIDERS["wordcab"]["options"] == {
            "diarize": True,
            "dual_channel": False,
            "only_api": False,
            "source": "audio",
        }
        assert PROVIDERS["wordcab"]["pricing"] == {
            "value": 0.00665,
            "unit": "minute",
        }

    def test_providers_wordcab_hosted(self) -> None:
        """Test Wordcab self-hosted provider."""
        assert PROVIDERS["wordcab-hosted"]["url"] == "http://{host}:{port}/api/v1/audio"
        assert PROVIDERS["wordcab-hosted"]["engine"] == "WordcabHosted"
        assert PROVIDERS["wordcab-hosted"]["output"] == "WordcabHostedOutput"
        assert PROVIDERS["wordcab-hosted"]["speaker_map"] == "WordcabHostedSpeakerMap"
        assert PROVIDERS["wordcab-hosted"]["concurrency_limit"] == 5
        assert PROVIDERS["wordcab-hosted"]["options"] == {
            "diarization": True,
            "dual_channel": False,
            "source_lang": "en",
        }
        assert PROVIDERS["wordcab-hosted"]["pricing"] == {
            "value": 0.0,
            "unit": "minute",
        }

    def test_metrics(self) -> None:
        """Test Metrics enum."""
        assert Metrics.__members__ == {
            "DER": "DER",
            "WER": "WER",
            "WRR": "WRR",
        }

        assert Metrics.DER == "DER"
        assert Metrics.WER == "WER"
        assert Metrics.WRR == "WRR"

        assert Metrics.DER == Metrics("DER")
        assert Metrics.WER == Metrics("WER")
        assert Metrics.WRR == Metrics("WRR")

        with pytest.raises(ValueError):
            Metrics("der")
        with pytest.raises(ValueError):
            Metrics("wer")
        with pytest.raises(ValueError):
            Metrics("wrr")
