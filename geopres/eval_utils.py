"""MTEB and intrinsic evaluation helpers.

The loss functions live in :mod:`geopres.losses` and are re-exported here so
that existing imports keep working.
"""

import json
import logging
import os

import mteb
import torch
from huggingface_hub import login
from mteb.cache import ResultCache
from sentence_transformers import SentenceTransformer

from geopres.config import embeddings_file, default_device
from geopres.losses import (
    compute_angular_loss,
    compute_positional_loss,
    compute_spearman_loss,
    geopres_loss,
)

__all__ = [
    "compute_angular_loss",
    "compute_positional_loss",
    "compute_spearman_loss",
    "geopres_loss",
    "eval_intrinsic",
    "evaluate_mteb",
    "evaluate_sts",
    "evaluate_retrieval",
    "evaluate_classification",
    "evaluate_clustering",
    "find_checkpoint_lowest_val_loss",
]

# Set random seed for reproducibility
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)


def login_if_token_available() -> None:
    """Log in to the Hugging Face Hub if ``HF_TOKEN`` is set (gated datasets)."""
    try:
        if hf_token := os.getenv("HF_TOKEN"):
            login(token=hf_token)
    except Exception as e:
        logger.warning(f"Could not log in to Hugging Face Hub: {e}")


def find_checkpoint_lowest_val_loss(trained_path: str) -> tuple[float, int]:
    """Find the checkpoint with the lowest validation loss."""
    checkpoint_dirs = [
        d for d in os.listdir(trained_path) if d.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoints found in {trained_path}"
        )

    last_checkpoint = max(
        int(d.split("checkpoint-")[-1]) for d in checkpoint_dirs
    )

    with open(
        os.path.join(
            trained_path, 
            f"checkpoint-{last_checkpoint}", 
            "trainer_state.json"
        )
    ) as f:
        trainer_state = json.load(f)
    
    lowest_val = 100000
    best_checkpoint = None
    for info_dict in trainer_state["log_history"]:
        if not "eval_loss" in info_dict:
            continue
        if info_dict["eval_loss"] < lowest_val:
            lowest_val = info_dict["eval_loss"]
            best_checkpoint = info_dict["step"]

    return lowest_val, best_checkpoint


# --- Eval functions ---
@torch.no_grad()
def eval_intrinsic(
    projection: torch.nn.Module,
    backbone_model_path: str,
    dataset_name = "sentence-paraphrases",
    checkpoint: str | None = None,
    cache_path: str | None = None,
    model_name: str | None = None,
    spearman_test_batch_size: int | None = 20000,
    spearman_local_or_global: str = "local"
):
    results = {
        "task_name": "IntrinsicEvaluation",
        "checkpoint": checkpoint,
        "spearman_loss": None,
        "angular_loss": None,
        "positional_loss": None,
    }

    test_embeddings_path = embeddings_file(dataset_name, backbone_model_path, "test")
    
    if not os.path.exists(test_embeddings_path):
        raise FileNotFoundError(
            f"Precalculated embeddings not found at {test_embeddings_path}"
        )
    
    high_dim_embeddings: torch.Tensor = torch.load(
        test_embeddings_path, weights_only=True
    ).to(default_device())
    low_dim_embeddings = projection(high_dim_embeddings)

    spearman_loss = compute_spearman_loss(
        low_dim_embeddings[:spearman_test_batch_size], 
        high_dim_embeddings[:spearman_test_batch_size], 
        training=False,
        weighted=False,
        local_or_global=spearman_local_or_global
    )
    results["spearman_loss"] = spearman_loss.item()

    angular_loss = compute_angular_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=False
    )
    results["angular_loss"] = angular_loss.item()

    positional_loss = compute_positional_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=False
    )
    results["positional_loss"] = positional_loss.item()

    logger.info(
        f"Intrinsic results for {model_name}, checkpoint {checkpoint}: "
        f"{json.dumps(results, indent=2)}"
    )
    if not cache_path:
        return results
    if not model_name:
        raise ValueError("model_name must be provided to save intrinsic test results")

    # Save in the same directory structure as MTEB results
    intrinsic_results_path = os.path.join(
        cache_path, 
        "results", 
        model_name.replace("/", "__"),
        "no_revision_available"
    )
    os.makedirs(intrinsic_results_path, exist_ok=True)
    save_path = os.path.join(intrinsic_results_path, "intrinsic.json")
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved at: {save_path}")

    return results
    

# --- MTEB evaluation ---

def evaluate_sts(
    model: SentenceTransformer, 
    cache_path: str,
    tasks_list: list[str] = [
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICK-R"
    ],
    languages: list[str] | None = None,
    batch_size: int = 4096,
    fast_mode: bool = False,
    overwrite_cache: bool = False
) -> float:
    """Evaluates a SentenceTransformer model on the STSBenchmark English dataset.

    Returns average Spearman correlation score.
    """
    if fast_mode:
        logger.info("Using fast mode for STS")
        tasks_list = ["STSBenchmark"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model, 
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing",
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)

def evaluate_retrieval(
    model: SentenceTransformer,
    cache_path: str,
    tasks_list: list[str] = [
        "ArguAna",

        # "MIRACLRetrievalHardNegatives",
        # "QuoraRetrievalHardNegatives",
        # "HotpotQAHardNegatives",
        # "DBPediaHardNegatives",
        # "NQHardNegatives",
        # "MSMARCOHardNegatives",

        "QuoraRetrieval",
        "HotpotQA",
        "NQ",
        "MSMARCO",
        "DBPedia",
    ],
    languages: list[str] | None = None,
    batch_size: int = 6,
    fast_mode: bool = False,
    overwrite_cache: bool = False
) -> float:
    """Evaluates a SentenceTransformer model on retrieval task.

    Returns average NDCG@10 score.
    """
    if fast_mode:
        logger.info("Using fast mode for retrieval")
        tasks_list = ["ArguAna"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model, 
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing"
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)
 

def evaluate_classification(
    model: SentenceTransformer,
    cache_path: str,
    tasks_list: list[str] = [
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "ImdbClassification",  # This has .v2 version
        "ToxicConversationsClassification",  # This has .v2 version
        "AmazonPolarityClassification",  # This has .v2 version    
    ],
    languages: list[str] | None = None,
    batch_size: int = 16,
    fast_mode: bool = False,
    overwrite_cache: bool = False
) -> float:
    """Evaluates a SentenceTransformer model on classification task.

    Returns accuracy score.
    """
    if fast_mode:
        logger.info("Using fast mode for classification")
        tasks_list = ["AmazonCounterfactualClassification"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model,
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing"
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)


def evaluate_clustering(
    model: SentenceTransformer,
    cache_path: str,
    tasks_list: list[str] = [
        "ArxivClusteringS2S",  # This has newer version (ArXivHierarchicalClusteringS2S)
        "RedditClustering",  # This has .v2 version
        "StackExchangeClustering"  # This has .v2 version
    ],
    languages: list[str] | None = None,
    batch_size: int = 20,
    fast_mode: bool = False,
    overwrite_cache: bool = False
):
    """Evaluates a SentenceTransformer model on clustering task.

    Returns average V-measure score.
    """
    if fast_mode:
        logger.info("Using fast mode for clustering")
        tasks_list = ["RedditClustering"]

    tasks = mteb.get_tasks(tasks=tasks_list, languages=languages, eval_splits=["test"])
    results = mteb.evaluate(
        tasks=tasks,
        model=model, 
        cache=ResultCache(cache_path=cache_path),
        encode_kwargs={'batch_size': batch_size},
        overwrite_strategy="always" if overwrite_cache else "only-missing"
    )
    test_scores = []
    for task_result in results:
        task_result_dict = task_result.to_dict()
        task_scores = [d["main_score"] for d in task_result_dict["scores"]["test"]]
        test_scores.append(sum(task_scores) / len(task_scores))
    return sum(test_scores) / len(test_scores)


def evaluate_mteb(
    model: SentenceTransformer,
    cache_path: str,
    sts_batch_size: int = 4096,
    retrieval_batch_size: int = 6,
    classification_batch_size: int = 16,
    clustering_batch_size: int = 20,
    skip_sts: bool = False,
    skip_retrieval: bool = False,
    skip_classification: bool = False,
    skip_clustering: bool = False,
    fast_mode: bool = False,
    overwrite_cache: bool = False
):
    """Evaluates a SentenceTransformer model on MTEB benchmark."""
    login_if_token_available()

    if not skip_sts:
        sts_score = evaluate_sts(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=sts_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Spearman correlation on STS test set: {sts_score:.4f}")

    if not skip_classification:
        classification_score = evaluate_classification(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=classification_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Classification accuracy score: {classification_score:.4f}")

    if not skip_clustering:
        clustering_score = evaluate_clustering(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=clustering_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Clustering V-measure score: {clustering_score:.4f}")

    if not skip_retrieval:
        retrieval_score = evaluate_retrieval(
            model=model,
            cache_path=cache_path,
            fast_mode=fast_mode,
            batch_size=retrieval_batch_size,
            overwrite_cache=overwrite_cache
        )
        logger.info(f"Retrieval NDCG@10 score: {retrieval_score:.4f}")
        
