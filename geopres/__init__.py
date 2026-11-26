"""GeoPres: post-hoc dimensionality reduction for text embeddings.

GeoPres learns a single linear map that compresses high-dimensional embeddings
while preserving pairwise Euclidean distances. Training and inference only need
PyTorch and, if you work with text, a sentence-transformers backbone.

Quick start::

    from geopres import GeoPres

    model = GeoPres("sentence-transformers/all-mpnet-base-v2", dim=32)
    model.fit(texts)
    reduced = model.embed(texts)          # (n, 32)
    model.save("geopres-mpnet-32")

    model = GeoPres.load_path("geopres-mpnet-32")
    reduced = model.embed(["new text"])

If you already have embeddings and only want to reduce them, skip the backbone
entirely::

    from geopres import Projection

    projection = Projection.from_pretrained("kkalyfo/geopres-mpnet-32")
    reduced = projection.project(vectors)   # (n, 768) -> (n, 32)

The training / evaluation pipeline of the paper (C4 pre-encoding, MTEB, the
baselines) lives in the sibling modules and is driven by the CLI entry points
(``geopres-train``, ``geopres-eval``, ``geopres-precalculate``).
"""

__version__ = "0.1.2"

from geopres.api import GeoPres, Projection
from geopres.encode import encode_corpus
from geopres.losses import (
    compute_angular_loss,
    compute_positional_loss,
    compute_spearman_loss,
    geopres_loss,
)
from geopres.projection import LinearProjection

__all__ = [
    "GeoPres",
    "LinearProjection",
    "Projection",
    "compute_angular_loss",
    "compute_positional_loss",
    "compute_spearman_loss",
    "encode_corpus",
    "geopres_loss",
    "__version__",
]
