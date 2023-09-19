"""Speaker map for RTASR."""

import string
from collections import OrderedDict
from enum import Enum
from typing import List


class AMISpeakerMap:
    """AMI speaker map."""

    def __init__(self, speaker_list: List[str]) -> None:
        """
        Initialize AMI speaker map.

        Args:
            speaker_list (List[str]):
                List of speakers to map from AMI to RTASR.

        Notes:
            The speaker list must keep the order of appearance in the audio file
            (i.e. the first speaker in the list is the first speaker in the audio
            file, the second speaker in the list is the second speaker in the audio
            file, etc.).
        """
        if not all(isinstance(speaker, str) for speaker in speaker_list):
            raise TypeError(
                "Speaker list must be a list of strings. "
                f"Got { {type(speaker) for speaker in speaker_list} }."
            )

        self.speakers: List[str] = list(OrderedDict.fromkeys(speaker_list).keys())

    def __getitem__(self, value: str) -> int:
        """Get speaker ID from speaker name."""
        speaker_index = self.speakers.index(value)
        return string.ascii_uppercase[speaker_index]

    def from_value(self, speaker_id: str) -> str:
        """Get speaker ID from speaker name."""
        return str(self.__getitem__(speaker_id))


class AssemblyAISpeakerMap(str, Enum):
    """AssemblyAI speaker map."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"  # noqa: E741
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"  # noqa: E741
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"

    @classmethod
    def from_value(cls, speaker_id: str) -> str:
        """Get speaker map from a string value"""
        _speaker_id = speaker_id.upper() if isinstance(speaker_id, str) else speaker_id

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker name {speaker_id} not found in speaker map."
                "HINT: Speaker names are in the format `A` where `A` is a letter, "
                "between A and Z. For example, `A` or `Z` are valid speaker names."
            )
        else:
            return cls[_speaker_id].value


class AwsSpeakerMap(str, Enum):
    """AWS speaker map."""


class AzureSpeakerMap(str, Enum):
    """Azure speaker map."""


class DeepgramSpeakerMap(str, Enum):
    """Deepgram speaker map."""

    _0 = "A"
    _1 = "B"
    _2 = "C"
    _3 = "D"
    _4 = "E"
    _5 = "F"
    _6 = "G"
    _7 = "H"
    _8 = "I"
    _9 = "J"
    _10 = "K"
    _11 = "L"
    _12 = "M"
    _13 = "N"
    _14 = "O"
    _15 = "P"
    _16 = "Q"
    _17 = "R"
    _18 = "S"
    _19 = "T"
    _20 = "U"
    _21 = "V"
    _22 = "W"
    _23 = "X"
    _24 = "Y"
    _25 = "Z"

    @classmethod
    def from_value(cls, speaker_id: int) -> str:
        """Get speaker map from an integer value."""
        _speaker_id = f"_{speaker_id}"

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker ID {speaker_id} not found in speaker map."
                "HINT: Speaker IDs are in the format `XX` where `XX` is a number, "
                "between 00 and 25. For example, `00` or `25` are valid speaker IDs."
            )
        else:
            return cls[_speaker_id].value


class GoogleSpeakerMap(str, Enum):
    """Google speaker map."""


class RevAISpeakerMap(str, Enum):
    """RevAI speaker map."""

    _0 = "A"
    _1 = "B"
    _2 = "C"
    _3 = "D"
    _4 = "E"
    _5 = "F"
    _6 = "G"
    _7 = "H"
    _8 = "I"
    _9 = "J"
    _10 = "K"
    _11 = "L"
    _12 = "M"
    _13 = "N"
    _14 = "O"
    _15 = "P"
    _16 = "Q"
    _17 = "R"
    _18 = "S"
    _19 = "T"
    _20 = "U"
    _21 = "V"
    _22 = "W"
    _23 = "X"
    _24 = "Y"
    _25 = "Z"

    @classmethod
    def from_value(cls, speaker_id: int) -> str:
        """Get speaker map from an integer value."""
        _speaker_id = f"_{speaker_id}"

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker ID {speaker_id} not found in speaker map."
                "HINT: Speaker IDs are in the format `XX` where `XX` is a number, "
                "between 00 and 25. For example, `00` or `25` are valid speaker IDs."
            )
        else:
            return cls[_speaker_id].value


class SpeechmaticsSpeakerMap(str, Enum):
    """Speechmatics speaker map."""

    S1 = "A"
    S2 = "B"
    S3 = "C"
    S4 = "D"
    S5 = "E"
    S6 = "F"
    S7 = "G"
    S8 = "H"
    S9 = "I"
    S10 = "J"
    S11 = "K"
    S12 = "L"
    S13 = "M"
    S14 = "N"
    S15 = "O"
    S16 = "P"
    S17 = "Q"
    S18 = "R"
    S19 = "S"
    S20 = "T"
    S21 = "U"
    S22 = "V"
    S23 = "W"
    S24 = "X"
    S25 = "Y"
    S26 = "Z"

    @classmethod
    def from_value(cls, speaker_id: str) -> str:
        """Get speaker map from a string value."""
        _speaker_id = speaker_id.upper() if isinstance(speaker_id, str) else speaker_id

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker name {speaker_id} not found in speaker map.HINT: Speaker"
                " names are in the format `SXX` where `XX` is a number, between 00 and"
                " 25. For example, `S00` or `S25` are valid speaker names."
            )
        else:
            return cls[_speaker_id].value


class VoxConverseSpeakerMap(str, Enum):
    """VoxConverse speaker map."""

    spk00 = "A"
    spk01 = "B"
    spk02 = "C"
    spk03 = "D"
    spk04 = "E"
    spk05 = "F"
    spk06 = "G"
    spk07 = "H"
    spk08 = "I"
    spk09 = "J"
    spk10 = "K"
    spk11 = "L"
    spk12 = "M"
    spk13 = "N"
    spk14 = "O"
    spk15 = "P"
    spk16 = "Q"
    spk17 = "R"
    spk18 = "S"
    spk19 = "T"
    spk20 = "U"
    spk21 = "V"
    spk22 = "W"
    spk23 = "X"
    spk24 = "Y"
    spk25 = "Z"

    @classmethod
    def from_value(cls, speaker_id: str) -> int:
        """Get speaker map from string."""
        _speaker_id = speaker_id.lower() if isinstance(speaker_id, str) else speaker_id

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker name {speaker_id} not found in VoxConverse speaker map.HINT:"
                " Speaker names are in the format `spkXX` where `XX` is a number,"
                " between 00 and 25. For example, `spk00` or `spk25` are valid speaker"
                " names."
            )
        else:
            return cls[_speaker_id].value


class WordcabSpeakerMap(str, Enum):
    """Wordcab speaker map."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"  # noqa: E741
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"  # noqa: E741
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"

    @classmethod
    def from_value(cls, speaker_id: str) -> str:
        """Get speaker map from a string value."""
        _speaker_id = speaker_id.upper() if isinstance(speaker_id, str) else speaker_id

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker name {speaker_id} not found in speaker map."
                "HINT: Speaker names are in the format `A` where `A` is a letter, "
                "between A and Z. For example, `A` or `Z` are valid speaker names."
            )
        else:
            return cls[_speaker_id].value


class WordcabHostedSpeakerMap(str, Enum):
    """Wordcab self-hosted speaker map."""

    _0 = "A"
    _1 = "B"
    _2 = "C"
    _3 = "D"
    _4 = "E"
    _5 = "F"
    _6 = "G"
    _7 = "H"
    _8 = "I"
    _9 = "J"
    _10 = "K"
    _11 = "L"
    _12 = "M"
    _13 = "N"
    _14 = "O"
    _15 = "P"
    _16 = "Q"
    _17 = "R"
    _18 = "S"
    _19 = "T"
    _20 = "U"
    _21 = "V"
    _22 = "W"
    _23 = "X"
    _24 = "Y"
    _25 = "Z"

    @classmethod
    def from_value(cls, speaker_id: int) -> str:
        """Get speaker map from an integer value."""
        _speaker_id = f"_{speaker_id}"

        if _speaker_id not in cls.__members__:
            raise ValueError(
                f"Speaker ID {speaker_id} not found in speaker map."
                "HINT: Speaker IDs are in the format `X` where `X` is a number, "
                "between 0 and 25. For example, `0` or `25` are valid speaker IDs."
            )
        else:
            return cls[_speaker_id].value
