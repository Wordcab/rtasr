"""Test the DER metric."""

from rtasr.evaluation.der import DerEvalMode


def test_der_mode() -> None:
    """Test the DER mode."""

    assert DerEvalMode.FULL == (0.0, False)
    assert DerEvalMode.FAIR == (0.25, False)
    assert DerEvalMode.FORGIVING == (0.25, True)
