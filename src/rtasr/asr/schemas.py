"""ASR providers schemas for handling request and response data."""

from typing import Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict


class ASROutput(BaseModel):
    """ASR output schema."""

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ASROutput":
        """Create an ASROutput object from a JSON object."""
        return cls(**data)


class AssemblyAIWord(BaseModel):
    """AssemblyAI word schema."""

    confidence: float
    end: float
    speaker: Union[str, None]
    start: float
    text: str


class AssemblyAIUtterance(BaseModel):
    """AssemblyAI utterance schema."""

    confidence: float
    end: float
    speaker: str
    start: float
    text: str
    words: List[AssemblyAIWord]


class AssemblyAIOutput(ASROutput):
    """AssemblyAI output schema."""

    acoustic_model: str
    audio_duration: int
    audio_url: str
    id: str
    language_model: str
    language_code: str
    status: str
    text: str
    utterances: Union[List[AssemblyAIUtterance], None] = None
    words: List[AssemblyAIWord]


class AwsOutput(ASROutput):
    """AWS output schema."""


class AzureOutput(ASROutput):
    """Azure output schema."""


class DeepgramWords(BaseModel):
    """Deepgram words schema."""

    confidence: float
    end: float
    punctuated_word: str
    speaker: int
    speaker_confidence: float
    start: float
    word: str


class DeepgramMetadata(BaseModel):
    """Deepgram metadata schema."""

    channels: int
    created: str
    duration: float
    models: List[str]
    model_info: Dict[str, Dict[str, str]]
    request_id: str
    sha256: str

    model_config = ConfigDict(protected_namespaces=())


class DeepgramAlternative(BaseModel):
    """Deepgram alternative schema."""

    confidence: float
    transcript: str
    words: List[DeepgramWords]


class DeepgramChannel(BaseModel):
    """Deepgram channel schema."""

    alternatives: List[DeepgramAlternative]


class DeepgramUtterance(BaseModel):
    """Deepgram utterance schema."""

    channel: int
    confidence: float
    end: float
    id: str
    start: float
    speaker: int
    transcript: str
    words: List[DeepgramWords]


class DeepgramResult(BaseModel):
    """Deepgram result schema."""

    channels: List[DeepgramChannel]
    utterances: List[DeepgramUtterance]


class DeepgramOutput(ASROutput):
    """Deepgram output schema."""

    metadata: DeepgramMetadata
    results: DeepgramResult


class GoogleOutput(ASROutput):
    """Google output schema."""


class RevAIElement(BaseModel):
    """RevAI element schema."""

    confidence: Union[float, None] = None
    end_ts: Union[float, None] = None
    ts: Union[float, None] = None
    type: str
    value: str


class RevAIMonologue(BaseModel):
    """RevAI monologue schema."""

    elements: List[RevAIElement]
    speaker: int


class RevAIOutput(ASROutput):
    """RevAI output schema."""

    monologues: List[RevAIMonologue]


class SpeechmaticsAlternative(BaseModel):
    """Speechmatics alternative schema."""

    confidence: float
    content: str
    language: str
    speaker: str


class SpeechmaticsJob(BaseModel):
    """Speechmatics job schema."""

    created_at: str
    data_name: str
    duration: float
    id: str


class SpeechmaticsMetadata(BaseModel):
    """Speechmatics metadata schema."""

    created_at: str
    language_pack_info: Dict[str, Union[str, bool]]
    transcription_config: Dict[str, str]
    type: str


class SpeechmaticsResult(BaseModel):
    """Speechmatics result schema."""

    alternatives: List[SpeechmaticsAlternative]
    attaches_to: Union[str, None] = None
    end_time: float
    is_eos: Union[bool, None] = None
    start_time: float
    type: str


class SpeechmaticsOutput(ASROutput):
    """Speechmatics output schema."""

    format: str
    job: SpeechmaticsJob
    metadata: SpeechmaticsMetadata
    results: List[SpeechmaticsResult]


class WordcabWord(BaseModel):
    """Wordcab word schema."""

    end: float
    score: float
    start: float
    word: str


class WordcabTranscript(BaseModel):
    """Wordcab transcript schema."""

    end: str
    speaker: str
    start: str
    text: str
    timestamp_end: int
    timestamp_start: int
    words: List[WordcabWord]


class WordcabOutput(ASROutput):
    """Wordcab output schema."""

    job_id_set: List[str]
    speaker_map: Dict[str, str]
    summary_id_set: List[str]
    transcript: List[WordcabTranscript]
    transcript_id: str


class WordcabHostedTranscript(BaseModel):
    """Wordcab hosted transcript schema."""

    end: float
    speaker: int
    start: float
    text: str
    words: List[WordcabWord]


class WordcabHostedProcessingTimes(BaseModel):
    """Wordcab hosted processing times schema."""

    total: float
    transcription: float
    diarization: float
    post_processing: float


class WordcabHostedOutput(ASROutput):
    """Wordcab hosted output schema."""

    audio_duration: float
    diarization: bool
    process_times: WordcabHostedProcessingTimes
    source_lang: str
    timestamps: str
    utterances: List[WordcabHostedTranscript]
