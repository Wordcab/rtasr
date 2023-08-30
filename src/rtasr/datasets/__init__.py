"""Datasets module for collecting datasets for benchmarking."""

from .dataset_ami import prepare_ami_dataset
from .dataset_fleurs import prepare_fleurs_dataset
from .dataset_voxconverse import prepare_voxconverse_dataset

__all__ = [
    "prepare_ami_dataset",
    "prepare_fleurs_dataset",
    "prepare_voxconverse_dataset",
]
