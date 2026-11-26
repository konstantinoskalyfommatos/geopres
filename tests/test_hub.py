"""Tests for the Hugging Face Hub helpers (no network access)."""

import numpy as np
import pytest
import torch

from geopres import GeoPres


@pytest.fixture
def fitted_model():
    generator = torch.Generator().manual_seed(2)
    embeddings = torch.randn(64, 32, generator=generator)
    model = GeoPres("sentence-transformers/all-mpnet-base-v2", dim=4, device="cpu")
    model.fit(embeddings, epochs=1, batch_size=32, verbose=False)
    return model, embeddings


def test_push_to_hub_requires_a_backbone_id():
    model = GeoPres(dim=4, device="cpu").fit(
        torch.randn(16, 8), epochs=1, batch_size=16, verbose=False
    )
    with pytest.raises(ValueError, match="backbone"):
        model.push_to_hub("kkalyfo/geopres-test")


def test_model_card_mentions_backbone_and_dim(fitted_model):
    model, _ = fitted_model
    card = model.model_card("kkalyfo/geopres-test")

    assert card.startswith("---")
    assert "library_name: geopres" in card
    assert "sentence-transformers/all-mpnet-base-v2" in card
    assert "GeoPres.from_pretrained(\"kkalyfo/geopres-test\")" in card
    assert "Projection.from_pretrained(\"kkalyfo/geopres-test\")" in card
    assert "https://huggingface.co/sentence-transformers/all-mpnet-base-v2" in card
    assert "https://openreview.net/forum?id=vAWo3CIzqp" in card
    assert "https://github.com/konstantinoskalyfommatos/geopres" in card
    assert "https://pypi.org/project/geopres/" in card


def test_from_pretrained_loads_the_downloaded_folder(fitted_model, tmp_path, monkeypatch):
    model, embeddings = fitted_model
    model.save(tmp_path / "repo")

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path / "repo")
    )

    loaded = GeoPres.from_pretrained("kkalyfo/geopres-test", device="cpu")

    assert loaded.dim == model.dim
    assert loaded.model_name == model.model_name
    np.testing.assert_allclose(
        loaded.embed(embeddings), model.embed(embeddings), atol=1e-6
    )
