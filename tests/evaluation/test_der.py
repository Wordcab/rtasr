"""Test the DER metric."""

from rtasr.evaluation.der import DEREvalMode


def test_der_mode() -> None:
    """Test the DER mode."""

    assert DEREvalMode.FULL == (0.0, False)
    assert DEREvalMode.FAIR == (0.25, False)
    assert DEREvalMode.FORGIVING == (0.25, True)
