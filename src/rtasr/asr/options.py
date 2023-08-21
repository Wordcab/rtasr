"""Options regroup all the ASR providers options classes."""

from typing import List, TypedDict

from typing_extensions import Literal


class AssemblyAIOptions(TypedDict, total=False):
    """
    The options for the AssemblyAI transcription.

    References from the AssemblyAI docs:
    https://www.assemblyai.com/docs/API%20reference/transcript#create-a-transcript
    """

    audio_url: str
    language_code: str
    punctuate: bool
    format_text: bool
    dual_channel: bool
    webhook_url: str
    webhook_auth_header_name: str
    webhook_auth_header_value: str
    audio_start_from: int
    audio_end_at: int
    word_boost: List[str]
    boost_param: Literal["low", "default", "high"]
    filter_profanity: bool
    redact_pii: bool
    redact_pii_audio: bool
    redact_pii_audio_quality: Literal["mp3", "wav"]
    redact_pii_policies: List[str]
    redact_pii_sub: Literal["entity_type", "hash"]
    speaker_labels: bool
    speakers_expected: int
    content_safety: bool
    iab_categories: bool
    custom_spelling: List[str]
    disfluencies: bool
    sentiment_analysis: bool
    auto_chapters: bool
    entity_detection: bool
    speech_threshold: float



class AwsOptions(TypedDict, total=False):
    """
    The options for the AWS transcription.

    References from the AWS docs:
    https://docs.aws.amazon.com/transcribe/latest/dg/API_StartTranscriptionJob.html
    """

    # TODO: Add the options.


class AzureOptions(TypedDict, total=False):
    """
    The options for the Azure transcription.

    References from the Azure docs:
    https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/rest-speech-to-text
    """

    # TODO: Add the options.


class DeepgramOptions(TypedDict, total=False):
    """
    The options for the Deepgram transcription.

    References from the Deepgram docs:
    https://developers.deepgram.com/documentation/features/
    """

    diarize: bool
    diarize_version: str
    language: str
    model: str
    multichannel: bool
    punctuate: bool
    tier: str
    utterances: bool
    version: str


class GoogleOptions(TypedDict, total=False):
    """
    The options for the Google transcription.

    References from the Google docs:
    https://cloud.google.com/speech-to-text/docs/reference/rest/v1/RecognitionConfig
    """

    # TODO: Add the options.


class RevAIOptions(TypedDict, total=False):
    """
    The options for the RevAI transcription.

    References from the Rev docs:
    https://www.rev.ai/docs/streaming
    """

    # TODO: Add the options.


class SpeechmaticsOptions(TypedDict, total=False):
    """
    The options for the Speechmatics transcription.

    References from the Speechmatics docs:
    """

    # TODO: Add the references from the Speechmatics docs.
    # TODO: Add the options.


class WordcabOptions(TypedDict, total=False):
    """
    The options for the Wordcab transcription.

    References from the Wordcab docs:
    https://docs.wordcab.com/reference/post_transcribe
    """

    alignment: bool
    diarization: bool
    display_name: str
    dual_channel: bool
    source_lang: str
