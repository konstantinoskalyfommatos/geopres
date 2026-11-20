"""Hugging Face ``Trainer`` used by the paper's training script.

This trainer exists to reproduce the paper: it adds checkpointing, early
stopping and step-based evaluation on top of the plain training loop that
:meth:`geopres.GeoPres.fit` uses. Both optimise the same loss
(:func:`geopres.losses.geopres_loss`).
"""

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments

from geopres.config import default_device
from geopres.losses import geopres_loss


torch.manual_seed(42)

logger = logging.getLogger(__name__)


class EmbeddingsDataset(Dataset):
    """A dataset wrapping a pre-computed embedding tensor."""

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class GeoPresTrainer(Trainer):
    """Trainer subclass for GeoPres dimensionality reduction."""

    default_training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=10,
        per_device_train_batch_size=20000,
        per_device_eval_batch_size=20000,
        weight_decay=0.1,
        eval_strategy="steps",
        eval_steps=100,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_drop_last=True,
        disable_tqdm=False,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        dataloader_pin_memory=True,
    )
    """To be used as training args unless specified."""

    def __init__(
        self,
        source_dim: int | None = None,
        target_dim: int | None = None,
        device: str | None = None,
        spearman: bool = False,
        spearman_local_or_global: str = "local",
        positional_loss_factor: float = 1.0,
        weighted_loss: bool = False,
        **kwargs,
    ):
        if kwargs.get("args") is None:
            kwargs["args"] = self.default_training_args

        # Auto-initialize model if not provided but dimensions are given
        if kwargs.get("model") is None:
            if source_dim is None or target_dim is None:
                raise ValueError(
                    "Either 'model' or both 'source_dim' and 'target_dim' "
                    "must be provided to GeoPresTrainer."
                )
            logger.info(
                "No model provided, initializing matrix mapping "
                f"R^{source_dim} to R^{target_dim}"
            )
            kwargs["model"] = nn.Linear(
                source_dim,
                target_dim,
                bias=False,
                device=device or default_device(),
            )

        super().__init__(**kwargs)
        self.data_collator = self.collate_embeddings
        self.positional_loss_factor = positional_loss_factor
        self.spearman = spearman
        self.spearman_local_or_global = spearman_local_or_global
        self.weighted_loss = weighted_loss

    def compute_loss(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """Distance preservation loss between projected and original embeddings."""
        high_dim_embeddings = inputs["input"]
        low_dim_embeddings = model(high_dim_embeddings)
        return geopres_loss(
            low_dim_embeddings=low_dim_embeddings,
            high_dim_embeddings=high_dim_embeddings,
            positional_loss_factor=self.positional_loss_factor,
            weighted=self.weighted_loss,
            spearman=self.spearman,
            spearman_local_or_global=self.spearman_local_or_global,
            training=True,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        metrics = self._evaluate_intrinsic(eval_dataset, metric_key_prefix)
        self.log(metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_intrinsic(self, eval_dataset, metric_key_prefix="eval"):
        """Returns the intrinsic evaluation loss on the validation dataset."""
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        total_loss = 0.0
        num_samples = 0

        for features in eval_dataloader:
            high_dim_embeddings = features["input"] if isinstance(features, dict) else features
            high_dim_embeddings = high_dim_embeddings.to(self.args.device)
            low_dim_embeddings = model(high_dim_embeddings)

            loss = geopres_loss(
                low_dim_embeddings=low_dim_embeddings,
                high_dim_embeddings=high_dim_embeddings,
                positional_loss_factor=self.positional_loss_factor,
                weighted=self.weighted_loss,
                spearman=self.spearman,
                spearman_local_or_global=self.spearman_local_or_global,
                training=False,
            )

            total_loss += loss.item()
            num_samples += 1

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return {f"{metric_key_prefix}_loss": avg_loss}

    def get_eval_dataloader(self, eval_dataset):
        """Override to set drop_last=True for the eval dataloader.

        Since we care about preserving relationships between a batch of
        embeddings, we do not want to evaluate on a small number of embeddings.
        """
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    @staticmethod
    def collate_embeddings(features):
        return {"input": torch.stack(features, dim=0)}
