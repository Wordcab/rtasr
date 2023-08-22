"""Test schemas for ASR module."""

import pytest
from pydantic import BaseModel

from rtasr.asr.schemas import (
    ASROutput,
    AssemblyAIOutput,
    AssemblyAIUtterance,
    AssemblyAIWord,
    DeepgramAlternative,
    DeepgramChannel,
    DeepgramMetadata,
    DeepgramOutput,
    DeepgramResult,
    DeepgramUtterance,
    DeepgramWords,
    RevAIElement,
    RevAIMonologue,
    RevAIOutput,
    SpeechmaticsAlternative,
    SpeechmaticsJob,
    SpeechmaticsMetadata,
    SpeechmaticsOutput,
    SpeechmaticsResult,
    WordcabOutput,
    WordcabTranscript,
    WordcabWord,
)


class TestASROutputs:
    """Test ASR output schemas."""

    def test_asr_output(self):
        """Test ASR output schema."""
        output = ASROutput(
            engine="engine",
            metadata={"metadata": "metadata"},
            output={"output": "output"},
        )

        assert isinstance(output, ASROutput)
        assert isinstance(output, BaseModel)

        assert hasattr(output, "from_json")
        assert callable(output.from_json)

        assert not hasattr(output, "engine")
        assert not hasattr(output, "metadata")
        assert not hasattr(output, "output")

        with pytest.raises(AttributeError):
            assert output.engine == "engine"

        with pytest.raises(AttributeError):
            assert output.metadata == {"metadata": "metadata"}

        with pytest.raises(AttributeError):
            assert output.output == {"output": "output"}

    @pytest.mark.usefixtures("load_sample_file_assemblyai")
    def test_assemblyai_output(self, load_sample_file_assemblyai) -> None:
        """Test AssemblyAI output schema."""
        asr_output = AssemblyAIOutput.from_json(load_sample_file_assemblyai)

        assert isinstance(asr_output, AssemblyAIOutput)
        assert isinstance(asr_output, ASROutput)
        assert isinstance(asr_output, BaseModel)

        assert hasattr(asr_output, "acoustic_model")
        assert isinstance(asr_output.acoustic_model, str)

        assert hasattr(asr_output, "audio_duration")
        assert isinstance(asr_output.audio_duration, int)

        assert hasattr(asr_output, "audio_url")
        assert isinstance(asr_output.audio_url, str)

        assert hasattr(asr_output, "id")
        assert isinstance(asr_output.id, str)

        assert hasattr(asr_output, "language_model")
        assert isinstance(asr_output.language_model, str)

        assert hasattr(asr_output, "language_code")
        assert isinstance(asr_output.language_code, str)

        assert hasattr(asr_output, "status")
        assert isinstance(asr_output.status, str)

        assert hasattr(asr_output, "text")
        assert isinstance(asr_output.text, str)

        assert hasattr(asr_output, "utterances")
        assert isinstance(asr_output.utterances, (list, type(None)))
        if asr_output.utterances is not None:
            assert isinstance(asr_output.utterances[0], AssemblyAIUtterance)

        assert hasattr(asr_output, "words")
        assert isinstance(asr_output.words, list)
        assert isinstance(asr_output.words[0], AssemblyAIWord)

    @pytest.mark.usefixtures("load_sample_file_deepgram")
    def test_deepgram_output(self, load_sample_file_deepgram) -> None:
        """Test Deepgram output schema."""
        asr_output = DeepgramOutput.from_json(load_sample_file_deepgram)

        assert isinstance(asr_output, DeepgramOutput)
        assert isinstance(asr_output, ASROutput)
        assert isinstance(asr_output, BaseModel)

        assert hasattr(asr_output, "metadata")
        assert isinstance(asr_output.metadata, DeepgramMetadata)

        assert hasattr(asr_output.metadata, "channels")
        assert isinstance(asr_output.metadata.channels, int)

        assert hasattr(asr_output.metadata, "created")
        assert isinstance(asr_output.metadata.created, str)

        assert hasattr(asr_output.metadata, "duration")
        assert isinstance(asr_output.metadata.duration, float)

        assert hasattr(asr_output.metadata, "models")
        assert isinstance(asr_output.metadata.models, list)
        assert isinstance(asr_output.metadata.models[0], str)

        assert hasattr(asr_output.metadata, "model_info")
        assert isinstance(asr_output.metadata.model_info, dict)

        assert hasattr(asr_output.metadata, "request_id")
        assert isinstance(asr_output.metadata.request_id, str)

        assert hasattr(asr_output.metadata, "sha256")
        assert isinstance(asr_output.metadata.sha256, str)

        assert hasattr(asr_output, "results")
        assert isinstance(asr_output.results, DeepgramResult)

        assert hasattr(asr_output.results, "channels")
        assert isinstance(asr_output.results.channels, list)
        assert isinstance(asr_output.results.channels[0], DeepgramChannel)

        assert hasattr(asr_output.results, "utterances")
        assert isinstance(asr_output.results.utterances, list)
        assert isinstance(asr_output.results.utterances[0], DeepgramUtterance)

        assert hasattr(asr_output.results.channels[0], "alternatives")
        assert isinstance(asr_output.results.channels[0].alternatives, list)
        assert isinstance(
            asr_output.results.channels[0].alternatives[0], DeepgramAlternative
        )

        assert hasattr(asr_output.results.channels[0].alternatives[0], "confidence")
        assert isinstance(
            asr_output.results.channels[0].alternatives[0].confidence, float
        )

        assert hasattr(asr_output.results.channels[0].alternatives[0], "transcript")
        assert isinstance(
            asr_output.results.channels[0].alternatives[0].transcript, str
        )

        assert hasattr(asr_output.results.channels[0].alternatives[0], "words")
        assert isinstance(asr_output.results.channels[0].alternatives[0].words, list)
        assert isinstance(
            asr_output.results.channels[0].alternatives[0].words[0], DeepgramWords
        )

        words = asr_output.results.channels[0].alternatives[0].words[0]
        assert hasattr(words, "confidence")
        assert isinstance(words.confidence, float)

        assert hasattr(words, "end")
        assert isinstance(words.end, float)

        assert hasattr(words, "punctuated_word")
        assert isinstance(words.punctuated_word, str)

        assert hasattr(words, "speaker")
        assert isinstance(words.speaker, int)

        assert hasattr(words, "speaker_confidence")
        assert isinstance(words.speaker_confidence, float)

        assert hasattr(words, "start")
        assert isinstance(words.start, float)

        assert hasattr(words, "word")
        assert isinstance(words.word, str)

    @pytest.mark.usefixtures("load_sample_file_revai")
    def test_revai_output(self, load_sample_file_revai) -> None:
        """Test Rev.ai output schema."""
        asr_output = RevAIOutput.from_json(load_sample_file_revai)

        assert isinstance(asr_output, RevAIOutput)
        assert isinstance(asr_output, ASROutput)
        assert isinstance(asr_output, BaseModel)

        assert hasattr(asr_output, "monologues")
        assert isinstance(asr_output.monologues, list)
        assert isinstance(asr_output.monologues[0], RevAIMonologue)

        monologue = asr_output.monologues[0]

        assert hasattr(monologue, "elements")
        assert isinstance(monologue.elements, list)
        assert isinstance(monologue.elements[0], RevAIElement)

        assert hasattr(monologue, "speaker")
        assert isinstance(monologue.speaker, int)

        element = monologue.elements[0]

        assert hasattr(element, "confidence")
        assert isinstance(element.confidence, (float, type(None)))

        assert hasattr(element, "end_ts")
        assert isinstance(element.end_ts, (float, type(None)))

        assert hasattr(element, "ts")
        assert isinstance(element.ts, (float, type(None)))

        assert hasattr(element, "type")
        assert isinstance(element.type, str)

        assert hasattr(element, "value")
        assert isinstance(element.value, str)

    @pytest.mark.usefixtures("load_sample_file_speechmatics")
    def test_speechmatics_output(self, load_sample_file_speechmatics) -> None:
        """Test Speechmatics output schema."""
        asr_output = SpeechmaticsOutput.from_json(load_sample_file_speechmatics)

        assert isinstance(asr_output, SpeechmaticsOutput)
        assert isinstance(asr_output, ASROutput)
        assert isinstance(asr_output, BaseModel)

        assert hasattr(asr_output, "format")
        assert isinstance(asr_output.format, str)

        assert hasattr(asr_output, "job")
        assert isinstance(asr_output.job, SpeechmaticsJob)

        assert hasattr(asr_output, "metadata")
        assert isinstance(asr_output.metadata, SpeechmaticsMetadata)

        assert hasattr(asr_output, "results")
        assert isinstance(asr_output.results, list)
        assert isinstance(asr_output.results[0], SpeechmaticsResult)

        job = asr_output.job

        assert hasattr(job, "created_at")
        assert isinstance(job.created_at, str)

        assert hasattr(job, "data_name")
        assert isinstance(job.data_name, str)

        assert hasattr(job, "duration")
        assert isinstance(job.duration, float)

        assert hasattr(job, "id")
        assert isinstance(job.id, str)

        metadata = asr_output.metadata

        assert hasattr(metadata, "created_at")
        assert isinstance(metadata.created_at, str)

        assert hasattr(metadata, "language_pack_info")
        assert isinstance(metadata.language_pack_info, dict)

        assert hasattr(metadata, "transcription_config")
        assert isinstance(metadata.transcription_config, dict)

        assert hasattr(metadata, "type")
        assert isinstance(metadata.type, str)

        result = asr_output.results[0]

        assert hasattr(result, "alternatives")
        assert isinstance(result.alternatives, list)
        assert isinstance(result.alternatives[0], SpeechmaticsAlternative)

        assert hasattr(result, "attaches_to")
        assert isinstance(result.attaches_to, (str, type(None)))

        assert hasattr(result, "end_time")
        assert isinstance(result.end_time, float)

        assert hasattr(result, "is_eos")
        assert isinstance(result.is_eos, (bool, type(None)))

        assert hasattr(result, "start_time")
        assert isinstance(result.start_time, float)

        assert hasattr(result, "type")
        assert isinstance(result.type, str)

        alternative = result.alternatives[0]

        assert hasattr(alternative, "confidence")
        assert isinstance(alternative.confidence, float)

        assert hasattr(alternative, "content")
        assert isinstance(alternative.content, str)

        assert hasattr(alternative, "language")
        assert isinstance(alternative.language, str)

        assert hasattr(alternative, "speaker")
        assert isinstance(alternative.speaker, str)

    @pytest.mark.usefixtures("load_sample_file_wordcab")
    def test_wordcab_output(self, load_sample_file_wordcab) -> None:
        """Test Wordcab output schema."""
        asr_output = WordcabOutput.from_json(load_sample_file_wordcab)

        assert isinstance(asr_output, WordcabOutput)
        assert isinstance(asr_output, ASROutput)
        assert isinstance(asr_output, BaseModel)

        assert hasattr(asr_output, "job_id_set")
        assert isinstance(asr_output.job_id_set, list)
        assert isinstance(asr_output.job_id_set[0], str)

        assert hasattr(asr_output, "speaker_map")
        assert isinstance(asr_output.speaker_map, dict)

        assert hasattr(asr_output, "summary_id_set")
        assert isinstance(asr_output.summary_id_set, list)
        if asr_output.summary_id_set:
            assert isinstance(asr_output.summary_id_set[0], str)

        assert hasattr(asr_output, "transcript")
        assert isinstance(asr_output.transcript, list)
        assert isinstance(asr_output.transcript[0], WordcabTranscript)

        assert hasattr(asr_output, "transcript_id")
        assert isinstance(asr_output.transcript_id, str)

        transcript = asr_output.transcript[0]

        assert hasattr(transcript, "end")
        assert isinstance(transcript.end, str)

        assert hasattr(transcript, "speaker")
        assert isinstance(transcript.speaker, str)

        assert hasattr(transcript, "start")
        assert isinstance(transcript.start, str)

        assert hasattr(transcript, "text")
        assert isinstance(transcript.text, str)

        assert hasattr(transcript, "timestamp_end")
        assert isinstance(transcript.timestamp_end, int)

        assert hasattr(transcript, "timestamp_start")
        assert isinstance(transcript.timestamp_start, int)

        assert hasattr(transcript, "words")
        assert isinstance(transcript.words, list)
        assert isinstance(transcript.words[0], WordcabWord)

        word = transcript.words[0]

        assert hasattr(word, "end")
        assert isinstance(word.end, float)

        assert hasattr(word, "score")
        assert isinstance(word.score, float)

        assert hasattr(word, "start")
        assert isinstance(word.start, float)

        assert hasattr(word, "word")
        assert isinstance(word.word, str)
