"""The linear map that GeoPres learns: ``R^source_dim -> R^dim``."""

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class LinearProjection(nn.Linear):
    """A bias-free linear layer, i.e. the matrix ``W`` of the paper.

    It is a plain ``nn.Linear`` so that weights keep the default PyTorch
    initialisation (see the Johnson-Lindenstrauss discussion in the paper) and
    so that existing ``best_model.pt`` / ``model.safetensors`` checkpoints load
    unchanged.
    """

    def __init__(self, source_dim: int, dim: int):
        super().__init__(source_dim, dim, bias=False)

    @property
    def source_dim(self) -> int:
        return self.in_features

    @property
    def dim(self) -> int:
        return self.out_features

    def save(self, path: str) -> None:
        save_file({k: v.contiguous() for k, v in self.state_dict().items()}, path)

    @classmethod
    def load(
        cls,
        path: str,
        source_dim: int | None = None,
        dim: int | None = None,
        map_location: str = "cpu",
    ) -> "LinearProjection":
        """Load a projection from a ``.safetensors`` or ``.pt`` state dict.

        If the dimensions are not given they are read from the weight matrix,
        so usually you only need ``LinearProjection.load(path)``.
        """
        if path.endswith(".safetensors"):
            state_dict = load_file(path, device=map_location)
        else:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)

        weight = state_dict.get("weight")
        if weight is None:
            raise ValueError(
                f"{path} does not contain a 'weight' tensor. "
                f"Found keys: {sorted(state_dict)}"
            )

        source_dim = source_dim or weight.shape[1]
        dim = dim or weight.shape[0]

        projection = cls(source_dim, dim)
        projection.load_state_dict({"weight": weight})
        return projection
