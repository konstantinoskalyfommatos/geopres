"""Tests for the public API. No GPU, no network, no downloads."""

import numpy as np
import pytest
import torch

from geopres import GeoPres, Projection
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


# --------------------------------------------------------------- Projection


@pytest.fixture
def saved_projection(embeddings, tmp_path):
    model = GeoPres("sentence-transformers/all-mpnet-base-v2", dim=4, device="cpu")
    model.fit(embeddings, epochs=1, batch_size=64, verbose=False)
    model.save(tmp_path / "projection")
    return model, tmp_path / "projection"


def test_projection_matches_geopres_embed(saved_projection, embeddings):
    model, path = saved_projection
    projection = Projection.load_path(path)

    assert projection.dim == 4
    assert projection.source_dim == 32
    assert projection.backbone == "sentence-transformers/all-mpnet-base-v2"
    np.testing.assert_allclose(
        projection.project(embeddings.numpy()), model.embed(embeddings), atol=1e-5
    )


def test_projection_project_single_vector(saved_projection, embeddings):
    _, path = saved_projection
    projection = Projection.load_path(path)

    one = projection.project(embeddings[0].numpy())

    assert one.shape == (4,)
    np.testing.assert_allclose(one, projection.project(embeddings[:1])[0], atol=1e-6)


def test_projection_normalize(saved_projection, embeddings):
    _, path = saved_projection
    reduced = Projection.load_path(path).project(embeddings.numpy(), normalize=True)
    np.testing.assert_allclose(np.linalg.norm(reduced, axis=1), 1.0, atol=1e-5)


def test_projection_wrong_dimension_raises(saved_projection):
    _, path = saved_projection
    with pytest.raises(ValueError, match="32-dimensional"):
        Projection.load_path(path).project(np.zeros((10, 16), dtype=np.float32))


def test_projection_load_bare_state_dict(saved_projection, tmp_path):
    model, _ = saved_projection
    checkpoint = tmp_path / "best_model.pt"
    torch.save(model.projection.state_dict(), checkpoint)

    projection = Projection.load_path(checkpoint)

    assert projection.source_dim == 32
    assert projection.backbone is None  # no config.json next to a bare weight file


def test_projection_from_pretrained(saved_projection, monkeypatch):
    model, path = saved_projection

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(path))

    projection = Projection.from_pretrained("kkalyfo/geopres-test")

    assert projection.backbone == model.model_name


def test_projection_owns_its_weight():
    """Changing the caller's array must not change the projection."""
    weight = np.ones((4, 8), dtype=np.float32)
    projection = Projection(weight)

    weight[0, 0] = 99.0

    assert projection.weight[0, 0] == 1.0
    with pytest.raises(ValueError):
        projection.weight[0, 0] = 1.0


def test_projection_accepts_a_tensor_weight(embeddings):
    projection = Projection(torch.ones(4, 32))
    assert projection.project(embeddings.numpy()).shape == (300, 4)


def test_projection_empty_batch_raises(saved_projection):
    projection = Projection.load_path(saved_projection[1])

    for empty in ([], np.zeros((0, 32), dtype=np.float32), torch.zeros((0, 32))):
        with pytest.raises(ValueError, match="No embeddings"):
            projection.project(empty)


def test_projection_keeps_tensors_on_their_device(saved_projection, embeddings):
    """A tensor in must give a tensor out, with the same dtype (no host round trip)."""
    projection = Projection.load_path(saved_projection[1])

    out = projection.project(embeddings)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    np.testing.assert_allclose(out.numpy(), projection.project(embeddings.numpy()), atol=1e-5)

    double = projection.project(embeddings.double())
    assert double.dtype == torch.float64

    single = projection.project(embeddings[0])
    assert single.shape == (4,)
    np.testing.assert_allclose(single.numpy(), out[0].numpy(), atol=1e-5)


def test_projection_tensor_normalize(saved_projection, embeddings):
    projection = Projection.load_path(saved_projection[1])

    reduced = projection.project(embeddings, normalize=True)

    assert torch.allclose(reduced.norm(dim=1), torch.ones(len(embeddings)))


def test_to_torch_matches_project(saved_projection, embeddings):
    projection = Projection.load_path(saved_projection[1])

    linear = projection.to_torch()

    assert isinstance(linear, LinearProjection)
    assert linear.training is False
    assert linear.weight.requires_grad is False
    with torch.no_grad():
        np.testing.assert_allclose(
            linear(embeddings).numpy(), projection.project(embeddings.numpy()), atol=1e-5
        )


def test_embed_accepts_single_vector(embeddings):
    model = GeoPres(dim=4, device="cpu").fit(
        embeddings, epochs=1, batch_size=64, verbose=False
    )

    one = model.embed(embeddings[0].numpy())

    assert one.shape == (4,)
    np.testing.assert_allclose(one, model.embed(embeddings[:1])[0], atol=1e-6)


def test_embed_empty_batch_raises(embeddings):
    model = GeoPres(dim=4, device="cpu").fit(
        embeddings, epochs=1, batch_size=64, verbose=False
    )

    with pytest.raises(ValueError, match="No embeddings"):
        model.embed(np.zeros((0, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="No data"):
        model.embed([])


def test_projection_casts_non_float_tensors(saved_projection):
    """Integer tensors must not be matmul'd as integers."""
    projection = Projection.load_path(saved_projection[1])

    out = projection.project(torch.ones(2, 32, dtype=torch.int64))

    assert out.dtype == torch.float32
    np.testing.assert_allclose(
        out.numpy(), projection.project(torch.ones(2, 32)), atol=1e-5
    )
