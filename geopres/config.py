"""Paths and dtypes used by the research pipeline.

All artefacts live under ``<repository>/storage``::

    storage/
    |-- precalculated_embeddings/
    |-- trained_models/
    |-- trained_autoencoders/
    `-- evaluation_results/

The library API (``from geopres import GeoPres``) does not use any of this;
it is only needed by the training and evaluation scripts.
"""

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORAGE_PATH = str(PROJECT_ROOT / "storage")
EVALUATION_RESULTS_PATH = os.path.join(STORAGE_PATH, "evaluation_results")
TRAINED_MODELS_PATH = os.path.join(STORAGE_PATH, "trained_models")
TRAINED_AUTOENCODERS_PATH = os.path.join(STORAGE_PATH, "trained_autoencoders")

PRECALCULATED_EMBEDDINGS_PATH = os.path.join(STORAGE_PATH, "precalculated_embeddings")


def embeddings_file(dataset: str, backbone: str, split: str) -> str:
    """Path of one cached embedding tensor.

    Example: ``embeddings_file("c4", "jinaai/jina-embeddings-v2-small-en", "train")``
    -> ``$STORAGE_PATH/precalculated_embeddings/c4/jinaai__jina-embeddings-v2-small-en/train_embeddings.pt``
    """
    return os.path.join(
        PRECALCULATED_EMBEDDINGS_PATH,
        dataset.split("/")[-1],
        backbone.replace("/", "__"),
        f"{split}_embeddings.pt",
    )


def default_device() -> str:
    """``"cuda"`` if a GPU is visible, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


DTYPE_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "f16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fb16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "f32": "float32",
    "float": "float32",
    "fp64": "float64",
    "float64": "float64",
    "f64": "float64",
    "double": "float64",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
    "float8_e4m3fnuz": "float8_e4m3fnuz",
    "float8_e5m2fnuz": "float8_e5m2fnuz",
}


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapped = DTYPE_ALIASES.get(dtype_str, dtype_str)
    dtype = getattr(torch, mapped, None)
    if dtype is None:
        raise ValueError(
            f"Unknown dtype: '{dtype_str}'. "
            f"Supported: {', '.join(sorted(set(DTYPE_ALIASES.values())))}"
        )
    return dtype
