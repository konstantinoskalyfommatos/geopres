"""The public GeoPres API.

Typical use::

    from geopres import GeoPres

    model = GeoPres("sentence-transformers/all-mpnet-base-v2", dim=32)
    model.fit(texts)                    # or model.fit(precomputed_embeddings)
    reduced = model.embed(texts)        # (len(texts), 32) float32 numpy array
    model.save("geopres-mpnet-32")

    model = GeoPres.load_path("geopres-mpnet-32")
    reduced = model.embed(["some new text"])
"""

import json
import logging
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from geopres.config import default_device, parse_dtype
from geopres.encode import encode_texts
from geopres.losses import geopres_loss
from geopres.projection import LinearProjection

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# What the API accepts: a numpy array, a torch tensor, or plain python lists.
ArrayLike = np.ndarray | torch.Tensor
EmbeddingsLike = ArrayLike | Sequence[float] | Sequence[Sequence[float]]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "projection.safetensors"


def _ensure_logging() -> None:
    """Show progress lines even if the caller never configured logging."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _source_dim_error(
    shape: tuple[int, ...], source_dim: int, backbone: str | None
) -> ValueError:
    """The same error from every inference path, so the messages cannot drift."""
    return ValueError(
        f"Expected {source_dim}-dimensional embeddings, got shape {shape}. "
        f"This projection was trained for backbone {backbone!r}."
    )


def _empty_embeddings_error() -> ValueError:
    """An empty batch is a caller bug, not a result."""
    return ValueError("No embeddings given: expected at least one vector.")


def _read_projection(path: str | Path) -> tuple[LinearProjection, dict[str, Any]]:
    """Read the weights and ``config.json`` from a saved directory or a weight file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")

    config: dict[str, Any] = {}
    if path.is_dir():
        config_path = path / CONFIG_FILENAME
        if config_path.exists():
            config = json.loads(config_path.read_text())

        weights_path = path / WEIGHTS_FILENAME
        if not weights_path.exists():
            candidates = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No projection weights found in {path} "
                    f"(expected {WEIGHTS_FILENAME}, *.safetensors or *.pt)"
                )
            weights_path = candidates[0]
    else:
        weights_path = path

    return LinearProjection.load(str(weights_path)), config


def _download_projection(repo_id: str, revision: str | None, token: str | None) -> str:
    """Download the projection files of a Hub repo and return the local folder."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=[CONFIG_FILENAME, WEIGHTS_FILENAME, "*.safetensors"],
    )


class GeoPres:
    """Trains and applies a linear map that compresses embeddings.

    The map ``W`` (shape ``dim x source_dim``) is trained so that pairwise
    Euclidean distances between embeddings are preserved, i.e. so that
    ``||W y_i - W y_j|| ~ ||y_i - y_j||``. It is a single matrix multiply at
    inference time.

    Args:
        model: backbone used to turn text into embeddings. Either a
            sentence-transformers model id (loaded lazily, on first use) or an
            already loaded ``SentenceTransformer``. ``None`` is allowed when you
            only ever pass pre-computed embeddings.
        dim: target dimensionality.
        device: ``"cuda"``, ``"cpu"``, ... Defaults to CUDA when available.
        dtype: optional dtype for the backbone (e.g. ``torch.float16``).
        max_seq_length: optional ``max_seq_length`` override for the backbone.
        trust_remote_code: passed to ``SentenceTransformer``.
    """

    def __init__(
        self,
        model: "str | SentenceTransformer | None" = None,
        dim: int = 32,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_seq_length: int | None = None,
        trust_remote_code: bool = True,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        self.model = model
        self.dim = dim
        self.device = device or default_device()
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.trust_remote_code = trust_remote_code

        self.training_info: dict[str, Any] = {}
        self._backbone = None
        self._projection: LinearProjection | None = None

    # ------------------------------------------------------------------ state

    @property
    def model_name(self) -> str | None:
        """The backbone id, if the backbone was given as a string."""
        return self.model if isinstance(self.model, str) else None

    @property
    def backbone(self):
        """The loaded ``SentenceTransformer`` (loaded on first access)."""
        if self._backbone is None:
            if self.model is None:
                raise RuntimeError(
                    "This GeoPres instance has no backbone model, so it can only be "
                    "used on pre-computed embeddings. Pass a model id to "
                    "GeoPres(...) or to GeoPres.load_path(..., model=...)."
                )
            if isinstance(self.model, str):
                from geopres.encode import load_backbone

                logger.info(f"Loading backbone {self.model}")
                self._backbone = load_backbone(
                    self.model,
                    device=self.device,
                    dtype=self.dtype,
                    max_seq_length=self.max_seq_length,
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                self._backbone = self.model
        return self._backbone

    @property
    def projection(self) -> LinearProjection:
        """The trained ``dim x source_dim`` linear map.

        This is the torch module. For the numpy-only equivalent, see
        :class:`Projection` (``Projection(model.projection.weight)``).
        """
        if self._projection is None:
            raise RuntimeError(
                "No projection yet: call fit(...) first, or load a trained one "
                "with GeoPres.load_path(path)."
            )
        return self._projection

    @property
    def source_dim(self) -> int | None:
        """Dimensionality of the backbone embeddings, if known."""
        if self._projection is not None:
            return self._projection.source_dim
        if self._backbone is not None:
            return self._backbone.get_sentence_embedding_dimension()
        return None

    def __repr__(self) -> str:
        source = self.source_dim if self.source_dim is not None else "?"
        fitted = "fitted" if self._projection is not None else "not fitted"
        return (
            f"GeoPres(model={self.model_name!r}, "
            f"source_dim={source}, dim={self.dim}, {fitted})"
        )

    # ------------------------------------------------------------- data input

    def _to_embeddings(
        self,
        data: Sequence[str] | Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
        *,
        batch_size: int,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        """Turn texts or pre-computed embeddings into a float32 CPU tensor."""
        if isinstance(data, torch.Tensor):
            return data.detach().float().cpu()
        if isinstance(data, np.ndarray):
            return torch.as_tensor(data, dtype=torch.float32)
        if isinstance(data, str):
            raise TypeError(
                "Expected a list of strings or an array of embeddings, got a single string."
            )

        items = list(data)
        if not items:
            raise ValueError("No data given.")

        if isinstance(items[0], str):
            backbone = self.backbone  # raises early if there is no model
            logger.info(
                f"Encoding {len(items)} texts with {self.model_name or 'the backbone'}. "
                "This runs the model over every text, so it can take a while; "
                "pass pre-computed embeddings to skip this step (they can be re-used for" \
                "a future .fit call as well)."
            )
            return encode_texts(
                backbone,
                items,
                batch_size=batch_size,
                model_name=self.model_name,
                show_progress_bar=show_progress_bar,
            )
        return torch.as_tensor(np.asarray(items), dtype=torch.float32)

    # -------------------------------------------------------------- training

    def fit(
        self,
        data: Sequence[str] | Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
        *,
        epochs: int = 10,
        batch_size: int = 20000,
        learning_rate: float = 1e-2,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        validation_data: Sequence | np.ndarray | torch.Tensor | None = None,
        patience: int = 3,
        encode_batch_size: int = 32,
        seed: int = 42,
        verbose: bool = True,
    ) -> "GeoPres":
        """Train the projection.

        Args:
            data: texts (encoded with the backbone) or embeddings of shape
                ``(n, source_dim)``. The paper uses 10M C4 passages because
                results kept improving with more data, but that is not a
                requirement: a few hundred thousand samples are enough for a
                good projection.
            epochs: default 10, as in the paper.
            batch_size: embeddings per step. Defaults to the paper's 20 000
                (an RTX 3090 can handle this size). It is a hyperparameter, not just a memory setting:
                the loss compares all pairs inside a batch, so a smaller batch
                changes the training signal.
            learning_rate, weight_decay, warmup_ratio: optimizer settings
                (paper: 1e-2, 0.1, 0.1).
            validation_data: embeddings used to pick the best checkpoint and to
                stop early. If None, training runs all ``epochs`` and keeps the last weights.
            patience: stop after this many epochs without validation
                improvement. Ignored when there is no validation data.
            encode_batch_size: batch size used only when ``data`` /
                ``validation_data`` are texts that need encoding. Ignored when
                you pass embeddings, as they are already encoded.
            seed: random seed.
            verbose: log one line per epoch.

        Returns:
            ``self``, so calls can be chained.

        # Note:
            For RAM-restricted machines: the data does not have to fit in RAM. 
            Memory-map the file and ``fit`` reads only the batches it needs::

                embeddings = torch.load("train.pt", mmap=True, weights_only=True)
                model = GeoPres("all-mpnet-base-v2", dim=32)
                model.fit(embeddings)

            To encode a corpus only once and train several projections on it,
            use :func:`geopres.encode_corpus`.
        """
        if verbose:
            _ensure_logging()
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train = self._to_embeddings(
            data, batch_size=encode_batch_size, show_progress_bar=verbose
        )
        validation = (
            self._to_embeddings(
                validation_data, batch_size=encode_batch_size, show_progress_bar=False
            )
            if validation_data is not None
            else None
        )

        if train.ndim != 2:
            raise ValueError(f"Expected a 2-D array of embeddings, got shape {tuple(train.shape)}")

        if len(train) < 2:
            raise ValueError(f"Need at least 2 embeddings to train on, got {len(train)}")
        if validation is not None and len(validation) < 2:
            logger.warning(
                f"validation_data has {len(validation)} embedding(s), need at least 2. "
                "Ignoring it and keeping the last checkpoint."
            )
            validation = None
        if validation is None:
            logger.warning(
                "No validation_data given: training keeps the LAST checkpoint and "
                "never stops early. Pass validation_data=... to keep the weights "
                "with the lowest validation loss instead."
            )

        batch_size = min(batch_size, len(train))

        self._projection = LinearProjection(train.shape[1], self.dim).to(self.device)
        optimizer = torch.optim.AdamW(
            self._projection.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        steps_per_epoch = len(train) // batch_size
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _linear_warmup_decay(step, warmup_steps, total_steps)
        )

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            self._projection.train()
            epoch_loss = 0.0
            num_batches = 0

            order = torch.randperm(len(train))
            for start in range(0, len(train) - batch_size + 1, batch_size):
                batch = train[order[start:start + batch_size]].to(self.device)

                loss = geopres_loss(
                    low_dim_embeddings=self._projection(batch),
                    high_dim_embeddings=batch,
                    training=True,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

            train_loss = epoch_loss / max(num_batches, 1)
            validation_loss = (
                self._validation_loss(validation, batch_size)
                if validation is not None
                else None
            )

            if verbose:
                message = f"epoch {epoch}/{epochs} - train_loss {train_loss:.6f}"
                if validation_loss is not None:
                    message += f" - validation_loss {validation_loss:.6f}"
                logger.info(message)

            if validation_loss is None:
                continue

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self._projection.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(no improvement for {patience} epochs)"
                        )
                    break

        if best_state is not None:
            self._projection.load_state_dict(best_state)

        self.training_info = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "positional_loss_factor": 1.0,
            "weighted_loss": False,
            "spearman": False,
            "seed": seed,
            "num_train_embeddings": int(len(train)),
            "checkpoint_selection": "best_validation" if best_state is not None else "last",
            "best_validation_loss": best_loss if best_state is not None else None,
        }
        return self

    @torch.no_grad()
    def _validation_loss(self, embeddings: torch.Tensor, batch_size: int) -> float:
        self._projection.eval()
        total_loss = 0.0
        total_items = 0
        for start in range(0, len(embeddings), batch_size):
            chunk = embeddings[start:start + batch_size].to(self.device)
            if len(chunk) < 2:
                continue
            loss = geopres_loss(
                low_dim_embeddings=self._projection(chunk),
                high_dim_embeddings=chunk,
                training=False,
            )
            total_loss += loss.item() * len(chunk)
            total_items += len(chunk)
        return total_loss / total_items if total_items else float("inf")

    # -------------------------------------------------------------- inference

    def embed(
        self,
        data: Sequence[str] | Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
        *,
        batch_size: int = 32,
        normalize: bool = False,
        as_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Project texts or high-dimensional embeddings into ``dim`` dimensions.

        Args:
            data: texts, one embedding of shape ``(source_dim,)``, or a batch of
                shape ``(n, source_dim)``.
            batch_size: batch size for encoding and projecting.
            normalize: L2-normalise the output (the projection does not
                preserve norms exactly).
            as_numpy: return a numpy array (default) instead of a tensor.

        Returns:
            Array of shape ``(n, dim)``, or ``(dim,)`` for a single embedding.

        Raises:
            ValueError: on an empty batch or embeddings of the wrong dimension.
        """
        projection = self.projection
        embeddings = self._to_embeddings(
            data, batch_size=batch_size, show_progress_bar=show_progress_bar
        )
        if embeddings.numel() == 0:
            raise _empty_embeddings_error()

        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[None, :]
        if embeddings.ndim != 2 or embeddings.shape[1] != projection.source_dim:
            raise _source_dim_error(
                tuple(embeddings.shape), projection.source_dim, self.model_name
            )

        projection.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(embeddings), batch_size):
                chunk = embeddings[start:start + batch_size].to(
                    device=self.device, dtype=projection.weight.dtype
                )
                reduced = projection(chunk).float()
                if normalize:
                    reduced = torch.nn.functional.normalize(reduced, p=2, dim=1)
                chunks.append(reduced.cpu())

        result = torch.cat(chunks, dim=0)
        if single:
            result = result[0]
        return result.numpy() if as_numpy else result

    def to_sentence_transformer(
        self,
        *,
        model_name_or_path: str | None = None,
        custom_model_name: str | None = None,
    ):
        """Wrap backbone + projection as a ``SentenceTransformer``.

        Handy for MTEB evaluation or for code that expects the
        sentence-transformers API.
        """
        from geopres.reduced_sentence_transformer import ReducedSentenceTransformer

        model_name_or_path = model_name_or_path or self.model_name
        if model_name_or_path is None:
            raise ValueError(
                "Cannot build a SentenceTransformer without a backbone id; "
                "pass model_name_or_path=..."
            )
        return ReducedSentenceTransformer(
            model_name_or_path=model_name_or_path,
            projection=self.projection,
            output_dim=self.dim,
            device=self.device,
            custom_model_name=custom_model_name,
        )

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> str:
        """Save the projection to a directory (``config.json`` + weights)."""
        projection = self.projection
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        projection.save(str(path / WEIGHTS_FILENAME))

        config = {
            "format_version": FORMAT_VERSION,
            "backbone": self.model_name,
            "source_dim": projection.source_dim,
            "dim": projection.dim,
            "dtype": _dtype_name(self.dtype) if self.dtype is not None else None,
            "max_seq_length": self.max_seq_length,
            "training": self.training_info,
        }
        (path / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n")

        logger.info(f"Saved GeoPres projection to {path}")
        return str(path)

    @classmethod
    def load_path(
        cls,
        path: str | Path,
        *,
        model: "str | SentenceTransformer | None" = None,
        device: str | None = None,
    ) -> "GeoPres":
        """Load a projection saved with :meth:`save`.

        Args:
            path: directory written by :meth:`save`, or a single
                ``.safetensors`` / ``.pt`` file holding the weight matrix
                (e.g. a ``best_model.pt`` from ``train_model.py``).
            model: backbone to use for encoding texts. Defaults to the backbone
                stored in ``config.json``.
            device: device for the projection.
        """
        projection, config = _read_projection(path)
        if config.get("dim") and int(config["dim"]) != projection.dim:
            logger.warning(
                f"config.json says dim={config['dim']} but the weights are "
                f"{projection.dim}-dimensional; using the weights."
            )

        dtype_name = config.get("dtype")
        instance = cls(
            model=model if model is not None else config.get("backbone"),
            dim=projection.dim,
            device=device,
            dtype=parse_dtype(dtype_name) if dtype_name else None,
            max_seq_length=config.get("max_seq_length"),
        )
        instance._projection = projection.to(instance.device)
        instance.training_info = config.get("training") or {}
        return instance


    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: str | None = None,
        commit_message: str = "Upload GeoPres projection",
        model_card: str | None = None,
    ) -> str:
        """Upload ``config.json``, the weights and a model card to the HF Hub.

        Args:
            repo_id: e.g. ``"kkalyfo/geopres-jina-v2-small-en-32"``.
            private: create the repo as private.
            token: HF token; defaults to the one in your environment / login.
            commit_message: commit message for the upload.
            model_card: full ``README.md`` to upload. Defaults to
                :meth:`model_card`.

        Returns:
            The URL of the model repository.
        """
        from huggingface_hub import HfApi

        if self.model_name is None:
            raise ValueError(
                "push_to_hub needs the backbone id, so the uploaded model knows "
                "which encoder it belongs to. Create the instance with "
                "GeoPres('<hf-model-id>', dim=...) and fit it."
            )

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "projection"
            self.save(folder)
            (folder / "README.md").write_text(model_card or self.model_card(repo_id))
            api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                commit_message=commit_message,
            )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Uploaded GeoPres projection to {url}")
        return url

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        model: "str | SentenceTransformer | None" = None,
        device: str | None = None,
        token: str | None = None,
    ) -> "GeoPres":
        """Load a projection from the HF Hub, e.g. ``"kkalyfo/geopres-jina-32"``.

        Args:
            repo_id: repository on the Hub.
            revision: branch, tag or commit hash.
            model: backbone to use for encoding texts. Defaults to the backbone
                stored in the repository's ``config.json``.
            device: device for the projection.
            token: HF token, for private repositories.
        """
        local_dir = _download_projection(repo_id, revision, token)
        return cls.load_path(local_dir, model=model, device=device)

    def model_card(self, repo_id: str) -> str:
        """Default model card for the Hub, used by :meth:`push_to_hub`."""
        projection = self.projection
        return f"""---
library_name: geopres
tags:
- geopres
- dimensionality-reduction
- embeddings
---

# GeoPres projection: {self.model_name} -> {self.dim} dimensions

Linear map `W` ({projection.dim} x {projection.source_dim}) trained with
[GeoPres](https://github.com/konstantinoskalyfommatos/geopres) to preserve
pairwise Euclidean distances between `{self.model_name}` embeddings.

Backbone (frozen, not included): [{self.model_name}](https://huggingface.co/{self.model_name})

Paper: [On Post-Hoc Dimensionality Reduction of Text Embeddings](https://openreview.net/forum?id=vAWo3CIzqp) (EACL 2027)

## Encode text and reduce it

```python
from geopres import GeoPres

model = GeoPres.from_pretrained("{repo_id}")
reduced = model.embed(["some text"])   # (1, {self.dim})
```

## Reduce embeddings you already have

If your embeddings come from somewhere else (TEI, a vector database, a batch
job), use the matrix on its own. Nothing is encoded and the backbone is not
downloaded:

```python
from geopres import Projection

projection = Projection.from_pretrained("{repo_id}")
reduced = projection.project(embeddings)   # (n, {projection.source_dim}) -> (n, {projection.dim})
```

The input must be `{self.model_name}` embeddings, computed with the same
pooling, normalisation and prefixes as the training data.

The backbone is frozen; inference is one matrix multiply.
"""


class Projection:
    """A trained GeoPres projection on its own: a matrix, no backbone.

    Use this when you already have embeddings and only want to reduce them, e.g.
    vectors produced by Hugging Face's TEI, a vector database or another
    service. Nothing is encoded, and no backbone is loaded or downloaded.

    The embeddings must come from the model the projection was trained on, with
    the same pooling, normalisation and (for instruction-based models such as
    Qwen3-Embedding) the same query/passage prefixes.

    Numpy in, numpy out; a tensor in, a tensor out on the same device and with
    the same dtype. Vectors that already live on a GPU are therefore never
    copied to the host.

    This is the numpy counterpart of :class:`~geopres.projection.LinearProjection`
    (the ``nn.Module`` that ``GeoPres.projection`` returns). Use
    :meth:`to_torch` to get that module back.

    Args:
        weight: the ``(dim, source_dim)`` matrix. It is copied, and the copy is
            read-only, so later changes to the input cannot alter the projection.
        backbone: id of the embedding model it was trained for (informational).
        training_info: whatever ``config.json`` recorded about the run.
    """

    def __init__(
        self,
        weight: EmbeddingsLike,
        *,
        backbone: str | None = None,
        training_info: dict[str, Any] | None = None,
    ):
        if isinstance(weight, torch.Tensor):
            weight = weight.detach().float().cpu().numpy()

        # Own the memory: ``np.asarray`` would alias a float32 input, so a caller
        # changing its own array would silently change the projection.
        self.weight = np.array(weight, dtype=np.float32, copy=True)
        if self.weight.ndim != 2:
            raise ValueError(
                f"weight must be 2-D with shape (dim, source_dim), got {self.weight.shape}"
            )
        self.weight.flags.writeable = False

        self.backbone = backbone
        self.training_info = dict(training_info or {})
        # The matrix as a tensor, per (device, dtype), so a GPU caller does not
        # copy it to the device again on every call.
        self._torch_weights: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    @property
    def dim(self) -> int:
        """Target dimensionality."""
        return self.weight.shape[0]

    @property
    def source_dim(self) -> int:
        """Dimensionality of the embeddings this projection expects."""
        return self.weight.shape[1]

    def __repr__(self) -> str:
        return (
            f"Projection(backbone={self.backbone!r}, "
            f"source_dim={self.source_dim}, dim={self.dim})"
        )

    def project(
        self, embeddings: EmbeddingsLike, *, normalize: bool = False
    ) -> np.ndarray | torch.Tensor:
        """Project one ``(source_dim,)`` vector or a ``(n, source_dim)`` batch.

        Args:
            embeddings: numpy array, list or tensor of embeddings.
            normalize: L2-normalise the output (the projection does not
                preserve norms exactly).

        Returns:
            ``(dim,)`` for a single vector, ``(n, dim)`` for a batch. A tensor
            input gives a tensor on the same device and dtype.

        Raises:
            ValueError: on an empty batch or embeddings of the wrong dimension.
        """
        if isinstance(embeddings, torch.Tensor):
            return self._project_torch(embeddings, normalize=normalize)

        vectors = np.asarray(embeddings, dtype=np.float32)
        given_shape = vectors.shape
        if vectors.size == 0:
            raise _empty_embeddings_error()

        single = vectors.ndim == 1
        if single:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[1] != self.source_dim:
            raise _source_dim_error(given_shape, self.source_dim, self.backbone)

        reduced = vectors @ self.weight.T
        if normalize:
            norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            reduced = reduced / np.maximum(norms, 1e-12)
        return reduced[0] if single else reduced

    def _project_torch(
        self, embeddings: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor:
        """Project a tensor without leaving its device or changing its dtype."""
        if not embeddings.is_floating_point():
            embeddings = embeddings.float()
        if embeddings.numel() == 0:
            raise _empty_embeddings_error()

        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.source_dim:
            raise _source_dim_error(
                tuple(embeddings.shape), self.source_dim, self.backbone
            )

        weight = self._torch_weight(embeddings.device, embeddings.dtype)
        reduced = embeddings @ weight.T
        if normalize:
            reduced = torch.nn.functional.normalize(reduced, p=2, dim=1)
        return reduced[0] if single else reduced

    def _torch_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        weight = self._torch_weights.get(key)
        if weight is None:
            weight = torch.tensor(self.weight, device=device, dtype=dtype)
            self._torch_weights[key] = weight
        return weight

    def to_torch(self, *, device: str | torch.device | None = None) -> LinearProjection:
        """Return the matrix as a frozen ``LinearProjection``.

        Useful to run the projection on a GPU, inside ``torch.compile``, or to
        export it (TorchScript, ONNX). :meth:`project` is faster on CPU.
        """
        linear = LinearProjection(self.source_dim, self.dim)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(self.weight))
        if device is not None:
            linear = linear.to(device)
        return linear.eval().requires_grad_(False)

    @classmethod
    def load_path(cls, path: str | Path) -> "Projection":
        """Load a directory written by :meth:`GeoPres.save`, or a bare weight file."""
        linear, config = _read_projection(path)
        if config.get("dim") and int(config["dim"]) != linear.dim:
            logger.warning(
                f"config.json says dim={config['dim']} but the weights are "
                f"{linear.dim}-dimensional; using the weights."
            )
        return cls(
            linear.weight.detach().cpu().numpy(),
            backbone=config.get("backbone"),
            training_info=config.get("training"),
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
    ) -> "Projection":
        """Load a projection from the HF Hub, e.g. ``"kkalyfo/geopres-mpnet-32"``."""
        return cls.load_path(_download_projection(repo_id, revision, token))


def _linear_warmup_decay(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup, then linear decay to zero (same shape as HF ``linear``)."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    remaining = max(0, total_steps - step)
    decay_steps = max(1, total_steps - warmup_steps)
    return remaining / decay_steps
