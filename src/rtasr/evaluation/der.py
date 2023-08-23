"""Diarization Error Rate (DER) evaluation implementation."""

from enum import Enum


class DEEvalMode(tuple, Enum):
    """The DER evaluation mode.

    Note:
        The evaluation mode is a tuple of (collar, ignore_overlap). There are
        three evaluation modes available:
        * FULL:
            the DIHARD challenge style evaluation, the most strict way of
            evaluating diarization (collar, ignore_overlap) = (0.0, False).
        * FAIR:
            the evaluation setup used in VoxSRC challenge, more permissive
            than the previous one (collar, ignore_overlap) = (0.25, False).
        * FORGIVING:
            the traditional evaluation setup, more permissive than the two
            previous ones (collar, ignore_overlap) = (0.25, True).

    Attributes:
        collar (float):
            The collar value to use.
        ignore_overlap (bool):
            Whether to ignore overlapped speech or not (i.e. speech segments
            that are not annotated as overlapped speech but that overlap with
            other speech segments).
    """

    FULL = (0.0, False)
    FAIR = (0.25, False)
    FORGIVING = (0.25, True)
