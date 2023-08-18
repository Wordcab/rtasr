"""ASR module regroups all the ASR related classes and functions."""

from .options import (
    AssemblyAIOptions,
    AwsOptions,
    AzureOptions,
    DeepgramOptions,
    GoogleOptions,
    RevAIOptions,
    SpeechmaticsOptions,
    WordcabOptions,
)
from .providers import (
    AssemblyAI,
    Aws,
    Azure,
    Deepgram,
    Google,
    RevAI,
    Speechmatics,
    Wordcab,
)

__all__ = [
    "AssemblyAI",
    "AssemblyAIOptions",
    "Aws",
    "AwsOptions",
    "Azure",
    "AzureOptions",
    "Deepgram",
    "DeepgramOptions",
    "Google",
    "GoogleOptions",
    "RevAI",
    "RevAIOptions",
    "Speechmatics",
    "SpeechmaticsOptions",
    "Wordcab",
    "WordcabOptions",
]
