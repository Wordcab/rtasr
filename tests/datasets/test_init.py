"""Test the __init__.py file in the datasets module."""

from rtasr.datasets import prepare_ami_dataset, prepare_voxconverse_dataset


def test_prepare_ami_dataset() -> None:
    """Test prepare_ami_dataset."""
    assert prepare_ami_dataset is not None
    assert callable(prepare_ami_dataset)


def test_prepare_voxconverse_dataset() -> None:
    """Test prepare_voxconverse_dataset."""
    assert prepare_voxconverse_dataset is not None
    assert callable(prepare_voxconverse_dataset)
