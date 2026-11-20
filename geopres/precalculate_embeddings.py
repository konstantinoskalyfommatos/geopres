"""Pre-encode a corpus with a backbone and cache the embeddings on disk.

Training a projection reads these cached tensors, so this is a one-time,
GPU-heavy step per backbone:

    geopres-precalculate --model_name jinaai/jina-embeddings-v2-small-en --train --val_test

Writes ``$STORAGE_PATH/precalculated_embeddings/<dataset>/<backbone>/<split>_embeddings.pt``.
"""

import logging
import os
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from tqdm import tqdm

from geopres.config import PRECALCULATED_EMBEDDINGS_PATH, default_device, parse_dtype
from geopres.encode import encode_texts, load_backbone

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_DATASET = "allenai/c4"
TRAIN_DATASET_CONFIG = "en"
VAL_TEST_DATASET = "agentlans/sentence-paraphrases"


def _output_dir(dataset_path: str, model_name: str) -> str:
    return os.path.join(
        PRECALCULATED_EMBEDDINGS_PATH,
        dataset_path.split("/")[-1],
        model_name.replace("/", "__"),
    )


def precalculate_train_embeddings(
    model_name: str,
    dataset_path: str,
    batch_size: int,
    text_column: str,
    dataset_name: str | None = None,
    max_examples: int = 2_500_000,
    encode_every: int = 1_000_000,
    dtype: torch.dtype | None = None,
    device: str | None = None,
) -> None:
    """Encode the training corpus, saving intermediate chunks as it goes.

    Chunks are written every ``encode_every`` examples so that a crashed or
    interrupted run can resume: existing chunks are skipped, and they are
    concatenated and deleted once all of them are on disk.
    """
    logger.info("Precalculating training embeddings")

    output_base_path = _output_dir(dataset_path, model_name)

    model = load_backbone(model_name, device=device, dtype=dtype)
    model.max_seq_length = 300

    ds_iter = load_dataset(
        dataset_path,
        dataset_name,
        split="train",
        streaming=True,
    )

    logger.info("Processing train dataset")

    total_seen = 0
    examples = []
    for ex in ds_iter:
        if total_seen > max_examples:
            break

        text = ex[text_column]
        # Keep small ones without tokenizing to save time
        if len(text.split()) < 200:
            continue
        examples.append(ex)
        total_seen += 1

        if total_seen % encode_every == 0:
            split_output_path = os.path.join(
                output_base_path, f"train_embeddings_{total_seen}.pt"
            )

            if os.path.exists(split_output_path):
                logger.info(
                    f"{split_output_path} already exists. Skipping encoding this batch"
                )
                examples = []
                continue

            logger.info(f"Encoding train examples. Total seen: {total_seen}")
            embeddings = encode_texts(
                model,
                [ex[text_column] for ex in examples],
                batch_size=batch_size,
                model_name=model_name,
            )

            os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
            torch.save(embeddings, split_output_path)
            logger.info(f"Saved train embeddings to {split_output_path}")

            examples = []

    logger.info("Merging saved train embeddings")
    all_embeddings = [
        torch.load(os.path.join(output_base_path, f"train_embeddings_{i}.pt"), weights_only=True)
        for i in range(encode_every, total_seen + 1, encode_every)
    ]
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, os.path.join(output_base_path, "train_embeddings.pt"))

    logger.info("Deleting intermediate embeddings")
    for i in range(encode_every, total_seen + 1, encode_every):
        os.remove(os.path.join(output_base_path, f"train_embeddings_{i}.pt"))

    logger.info("Finished precalculating train embeddings")


def precalculate_val_test_embeddings(
    model_name: str,
    dataset_path: str,
    batch_size: int,
    max_validation_examples: int = 10_000,
    max_test_examples: int = 10_000,
    dtype: torch.dtype | None = None,
    device: str | None = None,
) -> None:
    """Encode paraphrase pairs for validation and test.

    The dataset has a single ``validation`` split, so the first
    ``max_validation_examples`` pairs become the validation set and the next
    ``max_test_examples`` pairs become the test set (they do not overlap).
    """
    logger.info("Precalculating validation and test embeddings")

    output_base_path = _output_dir(dataset_path, model_name)

    model = load_backbone(model_name, device=device, dtype=dtype)
    model.max_seq_length = 512

    ds_iter = iter(load_dataset(dataset_path, split="validation"))

    for split, max_examples in zip(
        ["validation", "test"], [max_validation_examples, max_test_examples]
    ):
        logger.info(f"Processing split: {split}")

        examples = []
        total_seen = 0
        for ex in tqdm(ds_iter, desc="Loading examples"):
            text, paraphrase = ex["text"], ex["paraphrase"]
            # Keep small ones without tokenizing to save time
            if len(text.split()) < 350 and len(paraphrase.split()) < 350:
                examples += [text, paraphrase]
                total_seen += 1
            if total_seen >= max_examples:
                break

        logger.info(f"Filtered {split} dataset to {len(examples) / 2} paraphrase pairs")

        embeddings = encode_texts(
            model, examples, batch_size=batch_size, model_name=model_name
        )

        split_output_path = os.path.join(output_base_path, f"{split}_embeddings.pt")
        os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
        torch.save(embeddings, split_output_path)
        logger.info(f"Saved {split} embeddings to {split_output_path}")


def main() -> None:
    parser = ArgumentParser(description="Precalculate embeddings for a dataset")

    parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--backbone_dtype", type=str, default=None, help="dtype for backbone model (e.g., float16, bfloat16)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--train", action="store_true", help="Precalculate training embeddings")
    parser.add_argument("--val_test", action="store_true", help="Precalculate validation and test embeddings")
    parser.add_argument("--batch_size", type=int, default=8192, help="1024 works well for 24GB GPU memory and Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--max_train_examples", type=int, default=10_000_000, help="Maximum number of training examples to process")
    parser.add_argument("--max_val_examples", type=int, default=10_000, help="Maximum number of validation examples to process")
    parser.add_argument("--max_test_examples", type=int, default=10_000, help="Maximum number of test examples to process")
    parser.add_argument("--encode_every", type=int, default=100_000, help="Number of examples to encode before saving intermediate results")

    args = parser.parse_args()

    if not args.train and not args.val_test:
        parser.error("Provide at least one of --train or --val_test.")

    device = args.device or default_device()
    if device == "cpu":
        logger.warning(
            "Encoding 10M C4 passages on CPU is extremely slow. "
            "Use a GPU, or lower --max_train_examples."
        )

    dtype = parse_dtype(args.backbone_dtype) if args.backbone_dtype else None

    if args.train:
        precalculate_train_embeddings(
            model_name=args.model_name,
            dataset_path=TRAIN_DATASET,
            dataset_name=TRAIN_DATASET_CONFIG,
            batch_size=args.batch_size,
            text_column="text",
            max_examples=args.max_train_examples,
            encode_every=args.encode_every,
            dtype=dtype,
            device=device,
        )
    if args.val_test:
        precalculate_val_test_embeddings(
            model_name=args.model_name,
            dataset_path=VAL_TEST_DATASET,
            batch_size=args.batch_size,
            max_validation_examples=args.max_val_examples,
            max_test_examples=args.max_test_examples,
            dtype=dtype,
            device=device,
        )


if __name__ == "__main__":
    main()
