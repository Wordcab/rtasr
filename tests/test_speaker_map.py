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

        assert ami_map[speaker_map[0]] == "A"
        assert ami_map[speaker_map[1]] == "B"
        assert ami_map[speaker_map[2]] == "C"

        assert ami_map.from_value(speaker_map[0]) == "A"
        assert ami_map.from_value(speaker_map[1]) == "B"
        assert ami_map.from_value(speaker_map[2]) == "C"

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
            ("a", "A"),
            ("A", "A"),
            ("b", "B"),
            ("B", "B"),
            ("c", "C"),
            ("C", "C"),
            ("d", "D"),
            ("D", "D"),
            ("e", "E"),
            ("E", "E"),
            ("f", "F"),
            ("F", "F"),
            ("g", "G"),
            ("G", "G"),
            ("h", "H"),
            ("H", "H"),
            ("i", "I"),
            ("I", "I"),
            ("j", "J"),
            ("J", "J"),
            ("k", "K"),
            ("K", "K"),
            ("l", "L"),
            ("L", "L"),
            ("m", "M"),
            ("M", "M"),
            ("n", "N"),
            ("N", "N"),
            ("o", "O"),
            ("O", "O"),
            ("p", "P"),
            ("P", "P"),
            ("q", "Q"),
            ("Q", "Q"),
            ("r", "R"),
            ("R", "R"),
            ("s", "S"),
            ("S", "S"),
            ("t", "T"),
            ("T", "T"),
            ("u", "U"),
            ("U", "U"),
            ("v", "V"),
            ("V", "V"),
            ("w", "W"),
            ("W", "W"),
            ("x", "X"),
            ("X", "X"),
            ("y", "Y"),
            ("Y", "Y"),
            ("z", "Z"),
            ("Z", "Z"),
        ],
    )
    def test_letter_speaker_map_with_valid(self, speaker_id: str, value: str) -> None:
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
            (0, "A"),
            (1, "B"),
            (2, "C"),
            (3, "D"),
            (4, "E"),
            (5, "F"),
            (6, "G"),
            (7, "H"),
            (8, "I"),
            (9, "J"),
            (10, "K"),
            (11, "L"),
            (12, "M"),
            (13, "N"),
            (14, "O"),
            (15, "P"),
            (16, "Q"),
            (17, "R"),
            (18, "S"),
            (19, "T"),
            (20, "U"),
            (21, "V"),
            (22, "W"),
            (23, "X"),
            (24, "Y"),
            (25, "Z"),
        ],
    )
    def test_integer_speaker_map_with_valid(self, speaker_id: int, value: str) -> None:
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
            ("S1", "A"),
            ("S2", "B"),
            ("S3", "C"),
            ("S4", "D"),
            ("S5", "E"),
            ("S6", "F"),
            ("S7", "G"),
            ("S8", "H"),
            ("S9", "I"),
            ("S10", "J"),
            ("S11", "K"),
            ("S12", "L"),
            ("S13", "M"),
            ("S14", "N"),
            ("S15", "O"),
            ("S16", "P"),
            ("S17", "Q"),
            ("S18", "R"),
            ("S19", "S"),
            ("S20", "T"),
            ("S21", "U"),
            ("S22", "V"),
            ("S23", "W"),
            ("S24", "X"),
            ("S25", "Y"),
            ("S26", "Z"),
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
            ("spk00", "A"),
            ("spk01", "B"),
            ("spk02", "C"),
            ("spk03", "D"),
            ("spk04", "E"),
            ("spk05", "F"),
            ("spk06", "G"),
            ("spk07", "H"),
            ("spk08", "I"),
            ("spk09", "J"),
            ("spk10", "K"),
            ("spk11", "L"),
            ("spk12", "M"),
            ("spk13", "N"),
            ("spk14", "O"),
            ("spk15", "P"),
            ("spk16", "Q"),
            ("spk17", "R"),
            ("spk18", "S"),
            ("spk19", "T"),
            ("spk20", "U"),
            ("spk21", "V"),
            ("spk22", "W"),
            ("spk23", "X"),
            ("spk24", "Y"),
            ("spk25", "Z"),
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
