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
    ASRProvider,
    AssemblyAI,
    Aws,
    Azure,
    Deepgram,
    Google,
    ProviderResult,
    RevAI,
    Speechmatics,
    Wordcab,
)

__all__ = [
    "ASRProvider",
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
    "ProviderResult",
    "RevAI",
    "RevAIOptions",
    "Speechmatics",
    "SpeechmaticsOptions",
    "Wordcab",
    "WordcabOptions",
]
