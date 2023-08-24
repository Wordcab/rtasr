"""Speaker map for RTASR."""

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
        return self.speakers.index(value) + 1

    def from_value(self, speaker_id: str) -> int:
        """Get speaker ID from speaker name."""
        return self.__getitem__(speaker_id)


class AssemblyAISpeakerMap(int, Enum):
    """AssemblyAI speaker map."""

    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    G = 7
    H = 8
    I = 9  # noqa: E741
    J = 10
    K = 11
    L = 12
    M = 13
    N = 14
    O = 15  # noqa: E741
    P = 16
    Q = 17
    R = 18
    S = 19
    T = 20
    U = 21
    V = 22
    W = 23
    X = 24
    Y = 25
    Z = 26

    @classmethod
    def from_value(cls, speaker_id: str) -> int:
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


class AwsSpeakerMap(int, Enum):
    """AWS speaker map."""


class AzureSpeakerMap(int, Enum):
    """Azure speaker map."""


class DeepgramSpeakerMap(int, Enum):
    """Deepgram speaker map."""

    _0 = 1
    _1 = 2
    _2 = 3
    _3 = 4
    _4 = 5
    _5 = 6
    _6 = 7
    _7 = 8
    _8 = 9
    _9 = 10
    _10 = 11
    _11 = 12
    _12 = 13
    _13 = 14
    _14 = 15
    _15 = 16
    _16 = 17
    _17 = 18
    _18 = 19
    _19 = 20
    _20 = 21
    _21 = 22
    _22 = 23
    _23 = 24
    _24 = 25
    _25 = 26

    @classmethod
    def from_value(cls, speaker_id: int) -> int:
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


class GoogleSpeakerMap(int, Enum):
    """Google speaker map."""


class RevAISpeakerMap(int, Enum):
    """RevAI speaker map."""

    _0 = 1
    _1 = 2
    _2 = 3
    _3 = 4
    _4 = 5
    _5 = 6
    _6 = 7
    _7 = 8
    _8 = 9
    _9 = 10
    _10 = 11
    _11 = 12
    _12 = 13
    _13 = 14
    _14 = 15
    _15 = 16
    _16 = 17
    _17 = 18
    _18 = 19
    _19 = 20
    _20 = 21
    _21 = 22
    _22 = 23
    _23 = 24
    _24 = 25
    _25 = 26

    @classmethod
    def from_value(cls, speaker_id: int) -> int:
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


class SpeechmaticsSpeakerMap(int, Enum):
    """Speechmatics speaker map."""

    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4
    S5 = 5
    S6 = 6
    S7 = 7
    S8 = 8
    S9 = 9
    S10 = 10
    S11 = 11
    S12 = 12
    S13 = 13
    S14 = 14
    S15 = 15
    S16 = 16
    S17 = 17
    S18 = 18
    S19 = 19
    S20 = 20
    S21 = 21
    S22 = 22
    S23 = 23
    S24 = 24
    S25 = 25
    S26 = 26

    @classmethod
    def from_value(cls, speaker_id: str) -> int:
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


class VoxConverseSpeakerMap(int, Enum):
    """VoxConverse speaker map."""

    spk00 = 1
    spk01 = 2
    spk02 = 3
    spk03 = 4
    spk04 = 5
    spk05 = 6
    spk06 = 7
    spk07 = 8
    spk08 = 9
    spk09 = 10
    spk10 = 11
    spk11 = 12
    spk12 = 13
    spk13 = 14
    spk14 = 15
    spk15 = 16
    spk16 = 17
    spk17 = 18
    spk18 = 19
    spk19 = 20
    spk20 = 21
    spk21 = 22
    spk22 = 23
    spk23 = 24
    spk24 = 25
    spk25 = 26

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


class WordcabSpeakerMap(int, Enum):
    """Wordcab speaker map."""

    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    G = 7
    H = 8
    I = 9  # noqa: E741
    J = 10
    K = 11
    L = 12
    M = 13
    N = 14
    O = 15  # noqa: E741
    P = 16
    Q = 17
    R = 18
    S = 19
    T = 20
    U = 21
    V = 22
    W = 23
    X = 24
    Y = 25
    Z = 26

    @classmethod
    def from_value(cls, speaker_id: str) -> int:
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
