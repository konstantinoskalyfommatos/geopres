"""Loss functions used to train a GeoPres projection.

All functions are pure PyTorch and work on a single batch of embeddings:

- ``low_dim_embeddings``: batch projected by the linear map, shape ``(m, k)``
- ``high_dim_embeddings``: same batch in the original space, shape ``(m, n)``

The main loss is the positional (Euclidean distance preservation) loss of
Equation 1 in the paper. The angular and Spearman losses are the alternatives
studied in the appendix. ``torchsort`` is only needed for the Spearman losses
and is imported lazily, so it is an optional dependency.
"""

import torch


def compute_positional_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    weighted: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise Euclidean distance preservation loss (the main GeoPres loss).

    L = (1 / n) * sum_{i<j} w_ij * (d_low(x_i, x_j) - d_high(x_i, x_j))^2

    With ``weighted=False`` all pairs get weight 1. With ``weighted=True``
    close pairs get more weight, w_ij = 1 / (d_high(x_i, x_j) + eps).
    """
    low_dim_dist = torch.cdist(low_dim_embeddings, low_dim_embeddings, p=2)
    high_dim_dist = torch.cdist(high_dim_embeddings, high_dim_embeddings, p=2)

    n = low_dim_dist.size(0)
    triu_indices = torch.triu_indices(
        n, n, offset=1, device=low_dim_embeddings.device
    )

    low_d = low_dim_dist[triu_indices[0], triu_indices[1]]
    high_d = high_dim_dist[triu_indices[0], triu_indices[1]]

    if not weighted:
        loss = (low_d - high_d).pow(2).mean()
    else:
        weights = 1.0 / (high_d + eps)
        loss = (weights * (low_d - high_d).pow(2)).mean()

    return loss


def compute_angular_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    weighted: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise cosine similarity preservation loss."""
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)

    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    n = low_dim_sim.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=low_dim_embeddings.device)

    low_dim_sim_upper = low_dim_sim[triu_indices[0], triu_indices[1]]
    high_dim_sim_upper = high_dim_sim[triu_indices[0], triu_indices[1]]

    if not weighted:
        return (low_dim_sim_upper - high_dim_sim_upper).pow(2).mean()

    weights = 1.0 / (1.0 - high_dim_sim_upper + eps)
    return (weights * (low_dim_sim_upper - high_dim_sim_upper).pow(2)).mean()


# --- Spearman rank losses (appendix) ---

def _import_torchsort():
    try:
        import torchsort
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The Spearman losses need the optional 'torchsort' package. "
            "Install it with `pip install torchsort` (or `pip install geopres[spearman]`). "
            "The default GeoPres loss does not need it."
        ) from exc
    return torchsort


def spearmanr_differentiable(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False,
    **kw,
):
    """Differentiable Spearman correlation over all off-diagonal pairs."""
    torchsort = _import_torchsort()

    n = target.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=target.device)
    pred = pred[triu_indices[0], triu_indices[1]]
    target = target[triu_indices[0], triu_indices[1]]
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)

    pred_ranks = torchsort.soft_rank(pred, **kw)
    target_ranks = torchsort.soft_rank(target, **kw)

    if weighted:
        weights = target_ranks / target_ranks.sum(dim=1, keepdim=True)

        w_sum = weights.sum(dim=1, keepdim=True)
        pred_mean = (weights * pred_ranks).sum(dim=1, keepdim=True) / w_sum
        target_mean = (weights * target_ranks).sum(dim=1, keepdim=True) / w_sum

        pred_centered = pred_ranks - pred_mean
        target_centered = target_ranks - target_mean

        cov = (weights * pred_centered * target_centered).sum(dim=1)
        pred_var = (weights * pred_centered ** 2).sum(dim=1)
        target_var = (weights * target_centered ** 2).sum(dim=1)

        eps = 1e-8
        correlation = cov / (torch.sqrt(pred_var * target_var) + eps)
        return correlation.mean()

    pred_ranks = pred_ranks - pred_ranks.mean()
    pred_ranks = pred_ranks / pred_ranks.norm()
    target_ranks = target_ranks - target_ranks.mean()
    target_ranks = target_ranks / target_ranks.norm()
    return (pred_ranks * target_ranks).sum()


def spearmanr_differentiable_local(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False,
    **kw,
) -> torch.Tensor:
    """Differentiable Spearman correlation, computed per row of the similarity matrix."""
    torchsort = _import_torchsort()

    n = pred.shape[0]
    m = n - 1

    mask = ~torch.eye(n, dtype=torch.bool, device=pred.device)
    pred_offdiag = pred[mask].reshape(n, m)
    target_offdiag = target[mask].reshape(n, m)

    target_soft_ranks = torchsort.soft_rank(target_offdiag, **kw)

    if weighted:
        weights = target_soft_ranks / target_soft_ranks.sum(dim=1, keepdim=True)
    else:
        weights = torch.ones_like(target_soft_ranks)

    pred_ranks = torchsort.soft_rank(pred_offdiag, **kw)

    w_sum = weights.sum(dim=1, keepdim=True)
    pred_mean = (weights * pred_ranks).sum(dim=1, keepdim=True) / w_sum
    target_mean = (weights * target_soft_ranks).sum(dim=1, keepdim=True) / w_sum

    pred_centered = pred_ranks - pred_mean
    target_centered = target_soft_ranks - target_mean

    cov = (weights * pred_centered * target_centered).sum(dim=1)
    pred_var = (weights * pred_centered ** 2).sum(dim=1)
    target_var = (weights * target_centered ** 2).sum(dim=1)

    eps = 1e-8
    row_correlations = cov / (torch.sqrt(pred_var * target_var) + eps)

    return row_correlations.mean()


def compute_spearman_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    local_or_global: str = "local",
    training: bool = False,
    weighted: bool = False,
) -> torch.Tensor:
    """Spearman rank correlation loss between cosine similarity rankings."""
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)

    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    with torch.set_grad_enabled(training):
        if local_or_global == "local":
            return 1.0 - spearmanr_differentiable_local(
                low_dim_sim, high_dim_sim, weighted=weighted
            )
        return 1.0 - spearmanr_differentiable(
            low_dim_sim, high_dim_sim, weighted=weighted
        )


def geopres_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    positional_loss_factor: float = 1.0,
    weighted: bool = False,
    spearman: bool = False,
    spearman_local_or_global: str = "local",
    training: bool = True,
) -> torch.Tensor:
    """The loss that GeoPres actually optimizes.

    ``positional_loss_factor=1.0`` (the paper default) is the plain positional
    loss. Smaller values mix in the angular loss. ``spearman=True`` replaces
    both with the Spearman rank loss.
    """
    if spearman:
        return compute_spearman_loss(
            low_dim_embeddings=low_dim_embeddings,
            high_dim_embeddings=high_dim_embeddings,
            local_or_global=spearman_local_or_global,
            training=training,
            weighted=weighted,
        )

    loss = torch.zeros((), device=low_dim_embeddings.device)
    if positional_loss_factor > 0:
        loss = loss + positional_loss_factor * compute_positional_loss(
            low_dim_embeddings=low_dim_embeddings,
            high_dim_embeddings=high_dim_embeddings,
            weighted=weighted,
        )
    if positional_loss_factor < 1:
        loss = loss + (1 - positional_loss_factor) * compute_angular_loss(
            low_dim_embeddings=low_dim_embeddings,
            high_dim_embeddings=high_dim_embeddings,
            weighted=weighted,
        )
    return loss
