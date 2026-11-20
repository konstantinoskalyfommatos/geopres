"""Tests for the public API. No GPU, no network, no downloads."""

import numpy as np
import pytest
import torch

from geopres import GeoPres
from geopres.projection import LinearProjection


@pytest.fixture
def embeddings() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(300, 32, generator=generator)


def test_fit_and_embed_shapes(embeddings):
    model = GeoPres(dim=8, device="cpu")
    model.fit(embeddings, epochs=2, batch_size=64, verbose=False)

    reduced = model.embed(embeddings)

    assert reduced.shape == (300, 8)
    assert reduced.dtype == np.float32
    assert model.source_dim == 32
    assert model.dim == 8


def test_fit_reduces_distance_distortion(embeddings):
    """A trained projection should preserve pairwise distances better than an untrained one."""

    def mean_squared_distance_error(projection):
        with torch.no_grad():
            reduced = projection(embeddings)
        original_distances = torch.cdist(embeddings, embeddings)
        reduced_distances = torch.cdist(reduced, reduced)
        mask = torch.triu(torch.ones_like(original_distances, dtype=torch.bool), diagonal=1)
        return (original_distances[mask] - reduced_distances[mask]).pow(2).mean().item()

    untrained = LinearProjection(32, 8)
    before = mean_squared_distance_error(untrained)

    model = GeoPres(dim=8, device="cpu").fit(
        embeddings, epochs=20, batch_size=64, verbose=False
    )
    after = mean_squared_distance_error(model.projection)

    assert after < before


def test_embed_accepts_numpy(embeddings):
    model = GeoPres(dim=4, device="cpu").fit(embeddings, epochs=1, batch_size=64, verbose=False)
    assert model.embed(embeddings.numpy()).shape == (300, 4)
    assert model.embed(embeddings.numpy(), as_numpy=False).shape == (300, 4)


def test_normalize(embeddings):
    model = GeoPres(dim=4, device="cpu").fit(embeddings, epochs=1, batch_size=64, verbose=False)
    reduced = model.embed(embeddings, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(reduced, axis=1), 1.0, atol=1e-5)


def test_save_and_load_roundtrip(embeddings, tmp_path):
    model = GeoPres(dim=4, device="cpu").fit(embeddings, epochs=1, batch_size=64, verbose=False)
    model.save(tmp_path / "projection")

    loaded = GeoPres.load_path(tmp_path / "projection", device="cpu")

    assert loaded.dim == model.dim
    assert loaded.source_dim == model.source_dim
    np.testing.assert_allclose(
        loaded.embed(embeddings), model.embed(embeddings), atol=1e-6
    )


def test_load_bare_state_dict(embeddings, tmp_path):
    """``best_model.pt`` written by train_model.py must load."""
    model = GeoPres(dim=4, device="cpu").fit(embeddings, epochs=1, batch_size=64, verbose=False)
    checkpoint = tmp_path / "best_model.pt"
    torch.save(model.projection.state_dict(), checkpoint)

    loaded = GeoPres.load_path(checkpoint, device="cpu")

    np.testing.assert_allclose(
        loaded.embed(embeddings), model.embed(embeddings), atol=1e-6
    )


def test_embed_before_fit_raises(embeddings):
    with pytest.raises(RuntimeError, match="fit"):
        GeoPres(dim=4, device="cpu").embed(embeddings)


def test_embed_wrong_dimension_raises(embeddings):
    model = GeoPres(dim=4, device="cpu").fit(embeddings, epochs=1, batch_size=64, verbose=False)
    with pytest.raises(ValueError, match="32-dimensional"):
        model.embed(torch.randn(10, 16))


def test_fit_needs_two_embeddings():
    with pytest.raises(ValueError, match="at least 2"):
        GeoPres(dim=4, device="cpu").fit(torch.randn(1, 8), verbose=False)


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GeoPres.load_path(tmp_path / "nope")
