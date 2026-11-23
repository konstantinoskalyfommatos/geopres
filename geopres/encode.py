"""Encoding text with a backbone model, shared by the API and the scripts."""

import logging
from typing import TYPE_CHECKING

import torch

from geopres.config import default_device

if TYPE_CHECKING:  # sentence-transformers is only imported when it is needed
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Some backbones need non-default encode kwargs. Qwen3-Embedding applies an
# instruction prompt by default, which we do not want for plain passages.
MODEL_TO_ENCODE_KWARGS = {
    "Qwen/Qwen3-Embedding-0.6B": {"prompt": None},
}


def load_backbone(
    model_name_or_path: str,
    *,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    max_seq_length: int | None = None,
    trust_remote_code: bool = True,
    revision: str | None = None,
) -> "SentenceTransformer":
    """Load a frozen sentence-transformer backbone."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name_or_path,
        device=device or default_device(),
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    model.eval()
    return model


def encode_texts(
    model: "SentenceTransformer",
    texts: list[str],
    *,
    batch_size: int = 32,
    model_name: str | None = None,
    show_progress_bar: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Encode texts into a float32 CPU tensor of shape ``(len(texts), source_dim)``."""
    if not texts:
        raise ValueError("No texts to encode.")

    kwargs = {**MODEL_TO_ENCODE_KWARGS.get(model_name or "", {}), **kwargs}

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=show_progress_bar,
                batch_size=len(batch),
                **kwargs,
            )
            chunks.append(embeddings.float().cpu())
    return torch.cat(chunks, dim=0)


__all__ = ["MODEL_TO_ENCODE_KWARGS", "load_backbone", "encode_texts"]
