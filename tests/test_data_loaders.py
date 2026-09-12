import torch
from torch.utils.data import TensorDataset

from data.loaders import _deterministic_train_val_split


def test_train_validation_split_is_fixed_and_disjoint():
    dataset = TensorDataset(torch.arange(100))
    train_a, val_a = _deterministic_train_val_split(dataset, 0.1, 2026)
    train_b, val_b = _deterministic_train_val_split(dataset, 0.1, 2026)
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert len(train_a) == 90
    assert len(val_a) == 10
    assert set(train_a.indices).isdisjoint(val_a.indices)
    assert set(train_a.indices) | set(val_a.indices) == set(range(100))
