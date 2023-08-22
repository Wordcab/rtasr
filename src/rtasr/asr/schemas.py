"""ASR providers schemas for handling request and response data."""

from typing import Any, Dict, List

from pydantic import BaseModel


class ASROutput(BaseModel):
    """ASR output schema."""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ASROutput":
        """Create an ASROutput object from a JSON object."""
        return cls(**data)


class AssemblyAIOutput(ASROutput):
    """AssemblyAI output schema."""


class AwsOutput(ASROutput):
    """AWS output schema."""


class AzureOutput(ASROutput):
    """Azure output schema."""


class DeepgramOutput(ASROutput):
    """Deepgram output schema."""


class GoogleOutput(ASROutput):
    """Google output schema."""


class RevAIOutput(ASROutput):
    """RevAI output schema."""


class SpeechmaticsOutput(ASROutput):
    """Speechmatics output schema."""


class WordcabWord(BaseModel):
    """Wordcab word schema."""

    word: str
    start: float
    end: float
    score: float


class WordcabTranscript(BaseModel):
    """Wordcab transcript schema."""

    text: str
    start: str
    end: str
    speaker: str
    words: List[WordcabWord]
    timestamp_end: int
    timestamp_start: int


class WordcabOutput(ASROutput):
    """Wordcab output schema."""

    transcript_id: str
    job_id_set: List[str]
    summary_id_set: List[str]
    transcript: List[WordcabTranscript]
    speaker_map: Dict[str, str]
