"""Options regroup all the ASR providers options classes."""

from typing import Dict, List, TypedDict

from typing_extensions import Literal


class AssemblyAIOptions(TypedDict, total=False):
    """
    The options for the AssemblyAI transcription.

    References from the AssemblyAI docs:
    https://www.assemblyai.com/docs/API%20reference/transcript#create-a-transcript
    """

    audio_start_from: int
    audio_end_at: int
    audio_url: str
    auto_chapters: bool
    boost_param: Literal["low", "default", "high"]
    content_safety: bool
    custom_spelling: List[str]
    disfluencies: bool
    dual_channel: bool
    entity_detection: bool
    filter_profanity: bool
    format_text: bool
    iab_categories: bool
    language_code: str
    punctuate: bool
    redact_pii: bool
    redact_pii_audio: bool
    redact_pii_audio_quality: Literal["mp3", "wav"]
    redact_pii_policies: List[str]
    redact_pii_sub: Literal["entity_type", "hash"]
    sentiment_analysis: bool
    speakers_expected: int
    speaker_labels: bool
    speech_threshold: float
    webhook_url: str
    webhook_auth_header_name: str
    webhook_auth_header_value: str
    word_boost: List[str]


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
    https://docs.rev.ai/api/asynchronous/reference/#operation/SubmitTranscriptionJob
    """

    callback_url: str
    custom_vocabularies: List[str]
    custom_vocabulary_id: str
    delete_after_seconds: int
    filter_profanity: bool
    language: str
    metadata: str
    remove_disfluencies: bool
    skip_diarization: bool
    skip_postprocessing: bool
    skip_punctuation: bool
    speaker_channels_count: int
    transcriber: Literal["human", "machine"]
    verbatim: bool


class SpeechmaticsOptions(TypedDict, total=False):
    """
    The options for the Speechmatics transcription.

    References from the Speechmatics docs:
    https://docs.speechmatics.com/features
    """

    type: str
    transcription_config: Dict[str, str]


class WordcabOptions(TypedDict, total=False):
    """
    The options for the Wordcab transcription.

    References from the Wordcab docs:
    https://docs.wordcab.com/reference/post_transcribe
    """

    diarization: bool
    display_name: str
    dual_channel: bool
    source_lang: str


class WordcabHostedOptions(TypedDict, total=False):
    """
    The options for the Wordcab hosted transcription.

    References from the `wordcab-transcribe` project:
    https://github.com/Wordcab/wordcab-transcribe
    """

    diarization: bool
    dual_channel: bool
    source_lang: str
