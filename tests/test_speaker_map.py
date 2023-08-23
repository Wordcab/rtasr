"""Test speaker_map.py module."""

import re
from typing import Any, List

import pytest

from rtasr.speaker_map import (
    AMISpeakerMap,
    AssemblyAISpeakerMap,
    DeepgramSpeakerMap,
    RevAISpeakerMap,
    SpeechmaticsSpeakerMap,
    VoxConverseSpeakerMap,
    WordcabSpeakerMap,
)


class TestSpeakerMapping:
    """Test speaker mapping."""

    @pytest.mark.parametrize(
        "speaker_list, speaker_map",
        [
            (
                ["ABC", "DEF", "GHI", "GHI", "ABC", "DEF", "ABC"],
                ["ABC", "DEF", "GHI"],
            ),
            (["az37", "bc65", "de12", "az37", "bc65"], ["az37", "bc65", "de12"]),
            (["1", "2", "3"], ["1", "2", "3"]),
            (["1", "2", "3", "1", "2", "3"], ["1", "2", "3"]),
        ],
    )
    def test_ami_speaker_map_with_valid(
        self, speaker_list: List[str], speaker_map: List[str]
    ) -> None:
        """Test AMISpeakerMap."""
        ami_map = AMISpeakerMap(speaker_list)

        assert ami_map.speakers == speaker_map
        assert len(ami_map.speakers) == len(speaker_map)

        assert ami_map[speaker_map[0]] == 1
        assert ami_map[speaker_map[1]] == 2
        assert ami_map[speaker_map[2]] == 3

        assert ami_map.from_value(speaker_map[0]) == 1
        assert ami_map.from_value(speaker_map[1]) == 2
        assert ami_map.from_value(speaker_map[2]) == 3

    @pytest.mark.parametrize(
        "speaker_list",
        [
            ["ABC", 1, ["AD"]],
            ["ABC", 1, ["AD"], "DEF"],
            ["ABC", 1, ["AD"], "DEF", 2],
        ],
    )
    def test_ami_speaker_map_with_invalid(self, speaker_list: List[Any]) -> None:
        """Test AMISpeakerMap."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Speaker list must be a list of strings. "
                f"Got { {type(speaker) for speaker in speaker_list} }."
            ),
        ):
            AMISpeakerMap(speaker_list)

    @pytest.mark.parametrize(
        "speaker_id, value",
        [
            ("a", 1),
            ("A", 1),
            ("b", 2),
            ("B", 2),
            ("c", 3),
            ("C", 3),
            ("d", 4),
            ("D", 4),
            ("e", 5),
            ("E", 5),
            ("f", 6),
            ("F", 6),
            ("g", 7),
            ("G", 7),
            ("h", 8),
            ("H", 8),
            ("i", 9),
            ("I", 9),
            ("j", 10),
            ("J", 10),
            ("k", 11),
            ("K", 11),
            ("l", 12),
            ("L", 12),
            ("m", 13),
            ("M", 13),
            ("n", 14),
            ("N", 14),
            ("o", 15),
            ("O", 15),
            ("p", 16),
            ("P", 16),
            ("q", 17),
            ("Q", 17),
            ("r", 18),
            ("R", 18),
            ("s", 19),
            ("S", 19),
            ("t", 20),
            ("T", 20),
            ("u", 21),
            ("U", 21),
            ("v", 22),
            ("V", 22),
            ("w", 23),
            ("W", 23),
            ("x", 24),
            ("X", 24),
            ("y", 25),
            ("Y", 25),
            ("z", 26),
            ("Z", 26),
        ],
    )
    def test_letter_speaker_map_with_valid(self, speaker_id: str, value: int) -> None:
        """Test AssemblyAISpeakerMap and WordcabSpeakerMap with valid speaker ID."""
        assert AssemblyAISpeakerMap.from_value(speaker_id) == value
        assert WordcabSpeakerMap.from_value(speaker_id) == value
        assert AssemblyAISpeakerMap.from_value(
            speaker_id
        ) == WordcabSpeakerMap.from_value(speaker_id)

    @pytest.mark.parametrize("speaker_id", ["1", 1, "abc", "spk1", "SPEAKER A", "S21"])
    def test_letter_speaker_map_with_invalid(self, speaker_id: Any) -> None:
        """Test AssemblyAISpeakerMap and WordcabSpeakerMap with invalid speaker ID."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Speaker name {speaker_id} not found in speaker map."
                "HINT: Speaker names are in the format `A` where `A` is a letter, "
                "between A and Z. For example, `A` or `Z` are valid speaker names."
            ),
        ):
            AssemblyAISpeakerMap.from_value(speaker_id)

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Speaker name {speaker_id} not found in speaker map."
                "HINT: Speaker names are in the format `A` where `A` is a letter, "
                "between A and Z. For example, `A` or `Z` are valid speaker names."
            ),
        ):
            WordcabSpeakerMap.from_value(speaker_id)

    @pytest.mark.parametrize(
        "speaker_id, value",
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (24, 25),
            (25, 26),
        ],
    )
    def test_integer_speaker_map_with_valid(self, speaker_id: int, value: int) -> None:
        """Test DeepgramSpeakerMap and RevAISpeakerMap with valid speaker ID."""
        assert DeepgramSpeakerMap.from_value(speaker_id) == value
        assert RevAISpeakerMap.from_value(speaker_id) == value
        assert DeepgramSpeakerMap.from_value(speaker_id) == RevAISpeakerMap.from_value(
            speaker_id
        )

    @pytest.mark.parametrize(
        "speaker_id", ["-1", -1, 26, "26", "abc", "spk1", "SPEAKER A", "S21"]
    )
    def test_integer_speaker_map_with_invalid(self, speaker_id: Any) -> None:
        """Test DeepgramSpeakerMap and RevAISpeakerMap with invalid speaker ID."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Speaker ID {speaker_id} not found in speaker map."
                "HINT: Speaker IDs are in the format `XX` where `XX` is a number, "
                "between 00 and 25. For example, `00` or `25` are valid speaker IDs."
            ),
        ):
            DeepgramSpeakerMap.from_value(speaker_id)

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Speaker ID {speaker_id} not found in speaker map."
                "HINT: Speaker IDs are in the format `XX` where `XX` is a number, "
                "between 00 and 25. For example, `00` or `25` are valid speaker IDs."
            ),
        ):
            RevAISpeakerMap.from_value(speaker_id)

    @pytest.mark.parametrize(
        "speaker_id, value",
        [
            ("S1", 1),
            ("S2", 2),
            ("S3", 3),
            ("S4", 4),
            ("S5", 5),
            ("S6", 6),
            ("S7", 7),
            ("S8", 8),
            ("S9", 9),
            ("S10", 10),
            ("S11", 11),
            ("S12", 12),
            ("S13", 13),
            ("S14", 14),
            ("S15", 15),
            ("S16", 16),
            ("S17", 17),
            ("S18", 18),
            ("S19", 19),
            ("S20", 20),
            ("S21", 21),
            ("S22", 22),
            ("S23", 23),
            ("S24", 24),
            ("S25", 25),
            ("S26", 26),
        ],
    )
    def test_speechmatics_speaker_map_with_valid(
        self, speaker_id: str, value: int
    ) -> None:
        """Test SpeechmaticsSpeakerMap with valid speaker ID."""
        assert SpeechmaticsSpeakerMap.from_value(speaker_id) == value

    @pytest.mark.parametrize("speaker_id", ["1", 1, "abc", "spk1", "SPEAKER A", "S27"])
    def test_speechmatics_speaker_map_with_invalid(self, speaker_id: Any) -> None:
        """Test SpeechmaticsSpeakerMap with invalid speaker ID."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Speaker name {speaker_id} not found in speaker map.HINT: Speaker"
                " names are in the format `SXX` where `XX` is a number, between 00 and"
                " 25. For example, `S00` or `S25` are valid speaker names."
            ),
        ):
            SpeechmaticsSpeakerMap.from_value(speaker_id)

    @pytest.mark.parametrize(
        "speaker_id, value",
        [
            ("spk00", 1),
            ("spk01", 2),
            ("spk02", 3),
            ("spk03", 4),
            ("spk04", 5),
            ("spk05", 6),
            ("spk06", 7),
            ("spk07", 8),
            ("spk08", 9),
            ("spk09", 10),
            ("spk10", 11),
            ("spk11", 12),
            ("spk12", 13),
            ("spk13", 14),
            ("spk14", 15),
            ("spk15", 16),
            ("spk16", 17),
            ("spk17", 18),
            ("spk18", 19),
            ("spk19", 20),
            ("spk20", 21),
            ("spk21", 22),
            ("spk22", 23),
            ("spk23", 24),
            ("spk24", 25),
            ("spk25", 26),
        ],
    )
    def test_voxconverse_speaker_map_with_valid(
        self, speaker_id: str, value: int
    ) -> None:
        """Test VoxConverseSpeakerMap with valid speaker ID."""
        assert VoxConverseSpeakerMap.from_value(speaker_id) == value

    @pytest.mark.parametrize(
        "speaker_id", ["1", 1, "abc", "spk1", "SPEAKER A", "spk27"]
    )
    def test_voxconverse_speaker_map_with_invalid(self, speaker_id: Any) -> None:
        """Test VoxConverseSpeakerMap with invalid speaker ID."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Speaker name {speaker_id} not found in VoxConverse speaker map.HINT:"
                " Speaker names are in the format `spkXX` where `XX` is a number,"
                " between 00 and 25. For example, `spk00` or `spk25` are valid speaker"
                " names."
            ),
        ):
            VoxConverseSpeakerMap.from_value(speaker_id)
