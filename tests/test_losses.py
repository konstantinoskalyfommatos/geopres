"""Tests for the loss functions."""

import pytest
import torch

from geopres.losses import (
    compute_angular_loss,
    compute_positional_loss,
    geopres_loss,
)


@pytest.fixture
def batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(40, 12, generator=generator)


def test_positional_loss_is_zero_for_identical_spaces(batch):
    assert compute_positional_loss(batch, batch).item() == pytest.approx(0.0, abs=1e-6)


def test_positional_loss_positive_for_shrunk_space(batch):
    projection = torch.nn.Linear(12, 4, bias=False)
    assert compute_positional_loss(projection(batch), batch).item() > 0


def test_angular_loss_is_zero_for_identical_spaces(batch):
    assert compute_angular_loss(batch, batch).item() == pytest.approx(0.0, abs=1e-6)


def test_angular_loss_ignores_scale(batch):
    assert compute_angular_loss(3 * batch, batch).item() == pytest.approx(0.0, abs=1e-6)


def test_positional_loss_is_symmetric(batch):
    projection = torch.nn.Linear(12, 4, bias=False)
    reduced = projection(batch)
    assert compute_positional_loss(reduced, batch).item() == pytest.approx(
        compute_positional_loss(batch, reduced).item()
    )


def test_weighted_loss_puts_more_weight_on_close_pairs():
    close = torch.randn(30, 8) * 0.01
    plain = compute_positional_loss(close * 2, close, weighted=False)
    weighted = compute_positional_loss(close * 2, close, weighted=True)
    assert weighted > plain


def test_geopres_loss_default_is_positional(batch):
    projection = torch.nn.Linear(12, 4, bias=False)
    reduced = projection(batch)
    assert geopres_loss(reduced, batch).item() == pytest.approx(
        compute_positional_loss(reduced, batch).item()
    )


def test_geopres_loss_mixes_in_angular(batch):
    projection = torch.nn.Linear(12, 4, bias=False)
    reduced = projection(batch)
    mixed = geopres_loss(reduced, batch, positional_loss_factor=0.5)
    positional = compute_positional_loss(reduced, batch)
    angular = compute_angular_loss(reduced, batch)
    assert mixed.item() == pytest.approx(0.5 * positional.item() + 0.5 * angular.item())


def test_spearman_loss_needs_torchsort(batch):
    pytest.importorskip("torchsort")
    projection = torch.nn.Linear(12, 4, bias=False)
    loss = geopres_loss(projection(batch), batch, spearman=True, training=True)
    assert loss.item() >= 0
