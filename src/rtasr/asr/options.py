"""Options regroup all the ASR providers options classes."""

from typing import TypedDict

from typing_extensions import Literal


class AssemblyAIOptions(TypedDict, total=False):
    """
    The options for the AssemblyAI transcription.

    References from the AssemblyAI docs:
    """

    # TODO: Add the references from the AssemblyAI docs.
    # TODO: Add the options.


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

    diarize: Literal["false", "true"]
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
    """

    # TODO: Add the references from the Wordcab docs.
    # TODO: Add the options.
