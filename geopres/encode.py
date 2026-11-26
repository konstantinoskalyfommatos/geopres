"""Encoding text with a backbone model, shared by the API and the scripts."""

import logging
import os
from typing import TYPE_CHECKING, Sequence

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


def encode_corpus(
    texts: Sequence[str],
    path: str,
    model: "str | SentenceTransformer | None" = None,
    *,
    batch_size: int = 8192,
    chunk_size: int = 100_000,
    dtype: torch.dtype | None = None,
    device: str | None = None,
    overwrite: bool = False,
    show_progress_bar: bool = False,
) -> str:
    """Encode a corpus once and save it as one float32 tensor.

    Encoding is the expensive step, so do it once and train several projections
    on the result::

        encode_corpus(texts, "c4.pt", "sentence-transformers/all-mpnet-base-v2")
        model.fit(torch.load("c4.pt", mmap=True, weights_only=True))

    The corpus is encoded in ``chunk_size`` slices, each saved to
    ``<path>.part-NNNNN``. An existing part is not encoded again, so an
    interrupted run resumes where it stopped. The parts are concatenated into
    ``path`` at the end and then deleted.

    Args:
        texts: the texts to encode.
        path: output file, e.g. ``"c4.pt"``.
        model: a sentence-transformers id or an already loaded model.
        batch_size: texts per forward pass.
        chunk_size: texts per saved part (the resume granularity).
        dtype, device: where and in what precision to run the backbone.
        overwrite: re-encode even if ``path`` already exists.
        show_progress_bar: show a tqdm bar while encoding.

    Returns:
        ``path``, so it can be passed straight to ``torch.load``.
    """
    if isinstance(texts, (str, bytes)):
        raise TypeError("Expected a list of texts, got a single string.")
    texts = list(texts)
    if not texts:
        raise ValueError("No texts to encode.")
    if model is None:
        raise ValueError(
            "Pass model=<sentence-transformers id or loaded model> to encode_corpus."
        )
    if os.path.exists(path) and not overwrite:
        logger.info(f"{path} already exists, reusing it (pass overwrite=True to re-encode).")
        return path

    model_name = model if isinstance(model, str) else None
    backbone = load_backbone(model, device=device, dtype=dtype) if model_name else model

    total = len(texts)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    logger.info(
        f"Encoding {total} texts with {model_name or 'the given backbone'} into "
        f"{path}. This runs the model over every text and can take a while."
    )

    parts = []
    for index, start in enumerate(range(0, total, chunk_size)):
        part = f"{path}.part-{index:05d}"
        parts.append(part)
        if os.path.exists(part):
            logger.info(f"{part} already exists, skipping it")
            continue
        chunk = texts[start : start + chunk_size]
        torch.save(
            encode_texts(
                backbone,
                chunk,
                batch_size=batch_size,
                model_name=model_name,
                show_progress_bar=show_progress_bar,
            ),
            part,
        )
        logger.info(f"Saved {len(chunk)} embeddings to {part}")

    # Fill one preallocated tensor, so peak memory is the result plus one part.
    merged, offset = None, 0
    for part in parts:
        piece = torch.load(part, mmap=True, weights_only=True)
        if merged is None:
            merged = torch.empty((total, piece.shape[1]), dtype=torch.float32)
        merged[offset : offset + len(piece)] = piece
        offset += len(piece)
        del piece
    if offset != total:
        raise ValueError(
            f"The saved parts in {path}.part-* hold {offset} embeddings, expected "
            f"{total}. Delete them or pass overwrite=True."
        )

    torch.save(merged, path)
    for part in parts:
        os.remove(part)
    logger.info(f"Saved {tuple(merged.shape)} embeddings to {path}")
    return path


__all__ = ["MODEL_TO_ENCODE_KWARGS", "encode_corpus", "encode_texts", "load_backbone"]
