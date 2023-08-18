"""Options regroup all the ASR providers options classes."""

from typing import List, TypedDict

from typing_extensions import Literal


class DeepgramOptions(TypedDict, total=False):
    """
    The options for the Deepgram transcription.

    References from the Deepgram docs:
    https://developers.deepgram.com/documentation/features/
    """

    model: str
    version: str
    language: str
    punctuate: bool
    profanity_filter: bool
    diarize: Literal["false", "true"]
    diarize_version: str
    version: str
    multichannel: bool
    alternatives: int
    numbers: bool
    numbers_spaces: bool
    search: List[str]
    callback: str
    keywords: List[str]
    ner: str
    tier: str
    dates: bool
    date_format: str
    times: bool
    dictation: bool
    measurements: bool
    smart_format: bool
    replace: str
