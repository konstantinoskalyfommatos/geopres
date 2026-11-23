"""Evaluate a trained GeoPres projection on MTEB and on intrinsic metrics.

    geopres-eval --backbone_model jinaai/jina-embeddings-v2-small-en --target_dim 32
"""

import logging
import os
import sys
from argparse import ArgumentParser

import torch

from geopres.config import (
    EVALUATION_RESULTS_PATH,
    TRAINED_MODELS_PATH,
    default_device,
    parse_dtype,
)
from geopres.eval_utils import eval_intrinsic, evaluate_mteb, find_checkpoint_lowest_val_loss
from geopres.projection import LinearProjection
from geopres.reduced_sentence_transformer import ReducedSentenceTransformer

# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(
        description="Evaluate a reduced SentenceTransformer model on MTEB and intrinsic metrics"
    )
    parser.add_argument("--backbone_model", type=str, default="jinaai/jina-embeddings-v2-small-en", help="Name or path of the backbone SentenceTransformer model")
    parser.add_argument("--source_dim", default=512, type=int)
    parser.add_argument("--target_dim", type=int, default=32, help="Target dimension of the reduced embeddings")
    parser.add_argument("--train_batch_size", type=int, help="Batch size used for training", default=20000)
    parser.add_argument("--positional_loss_factor", type=float, default=1.0, help="factor for positional vs similarity loss")
    parser.add_argument("--skip_sts", action="store_true", help="Skip STS evaluation")
    parser.add_argument("--skip_classification", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--skip_retrieval", action="store_true", help="Skip retrieval evaluation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering evaluation")

    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite MTEB evaluation cache results")
    parser.add_argument("--fast_mode", action="store_true")

    parser.add_argument("--intrinsic_only", action="store_true", help="Intrinsic only")

    parser.add_argument("--sts_batch_size", type=int, default=2048, help="Batch size for STS evaluation")
    parser.add_argument("--retrieval_batch_size", type=int, default=6, help="Batch size for retrieval evaluation")
    parser.add_argument("--classification_batch_size", type=int, default=20, help="Batch size for classification evaluation")
    parser.add_argument("--clustering_batch_size", type=int, default=16, help="Batch size for clustering evaluation")
    parser.add_argument("--spearman_test_batch_size", type=int, default=20000, help="Batch size for intrinsic Spearman evaluation")

    parser.add_argument("--custom_suffix", type=str, default=None, help="Was added to the normal model name")
    parser.add_argument("--spearman", action="store_true", help="Differentiable Spearman correlation loss")
    parser.add_argument("--spearman_local_or_global", type=str, default="local", choices=["local", "global"], help="Whether to compute Spearman loss using local (only pairs in the same batch) or global (all pairs in the dataset) similarities")
    parser.add_argument("--backbone_dtype", type=str, default=None, help="dtype for backbone model (e.g., float16, bfloat16)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: cuda if available)")

    args = parser.parse_args()
    logger.info(f"Args: {args}")

    device = args.device or default_device()
    dtype = parse_dtype(args.backbone_dtype) if args.backbone_dtype else None

    if args.spearman:
        model_name = (
            f"{args.backbone_model}"
            f"_reduced_{args.target_dim}"
            f"_batch_{args.train_batch_size}"
            "_spearman"
            f"{'_' + args.custom_suffix if args.custom_suffix else ''}"
        )
    else:
        model_name = (
            f"{args.backbone_model}"
            f"_reduced_{args.target_dim}"
            f"_batch_{args.train_batch_size}"
            f"_poslossfactor_{args.positional_loss_factor}"
            f"{'_' + args.custom_suffix if args.custom_suffix else ''}"
        )

    trained_path = os.path.join(
        TRAINED_MODELS_PATH,
        args.backbone_model.replace("/", "__"),
        model_name.replace("/", "__"),
    )

    cache_path = os.path.join(
        EVALUATION_RESULTS_PATH,
        "trained_models",
        args.backbone_model.replace("/", "__"),
    )

    best_model_path = os.path.join(trained_path, "best_model.pt")
    logger.info(f"Loading best model from: {best_model_path}")
    projection_head = LinearProjection.load(
        best_model_path, source_dim=args.source_dim, dim=args.target_dim, map_location=device
    ).to(device)
    projection_head.eval()

    # Intrinsic evaluation
    logger.info("Evaluating intrinsic metrics on test set")
    eval_intrinsic(
        projection=projection_head,
        backbone_model_path=args.backbone_model,
        cache_path=cache_path,
        checkpoint=find_checkpoint_lowest_val_loss(trained_path)[1],
        model_name=model_name,
        spearman_test_batch_size=args.spearman_test_batch_size,
        spearman_local_or_global=args.spearman_local_or_global,
    )
    if args.intrinsic_only:
        sys.exit(0)

    # MTEB evaluation
    logger.info("Evaluating on MTEB benchmark")
    custom_model = ReducedSentenceTransformer(
        model_name_or_path=args.backbone_model,
        projection=projection_head,
        output_dim=args.target_dim,
        custom_model_name=model_name,
        dtype=dtype,
        device=device,
    )
    custom_model.eval()

    evaluate_mteb(
        model=custom_model,
        cache_path=cache_path,
        sts_batch_size=args.sts_batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        classification_batch_size=args.classification_batch_size,
        clustering_batch_size=args.clustering_batch_size,
        skip_sts=args.skip_sts,
        skip_retrieval=args.skip_retrieval,
        skip_classification=args.skip_classification,
        skip_clustering=args.skip_clustering,
        fast_mode=args.fast_mode,
        overwrite_cache=args.overwrite_cache,
    )


if __name__ == "__main__":
    main()
