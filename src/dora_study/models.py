from __future__ import annotations

import copy
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


AdapterMethod = Literal["frozen", "magnitude", "lora", "dora"]


class AdapterLinear(nn.Module):
    """Frozen linear layer with LoRA, DoRA, or magnitude-only adaptation.

    The DoRA path follows the published/Hugging Face optimization rule: the
    row-wise direction norm is detached from the gradient graph.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        method: AdapterMethod,
        rank: int = 0,
        alpha: float | None = None,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must be a 2D tensor")
        if method in {"lora", "dora"} and rank < 1:
            raise ValueError("rank must be positive for LoRA and DoRA")

        self.method = method
        self.rank = rank
        self.register_buffer("base_weight", weight.detach().clone())
        if bias is None:
            self.register_buffer("base_bias", None)
        else:
            self.register_buffer("base_bias", bias.detach().clone())

        out_features, in_features = weight.shape
        self.scaling = float(alpha if alpha is not None else rank) / max(rank, 1)

        if method in {"lora", "dora"}:
            generator = torch.Generator(device=weight.device).manual_seed(init_seed)
            a = torch.empty(rank, in_features, device=weight.device, dtype=weight.dtype)
            a.uniform_(-1.0 / math.sqrt(in_features), 1.0 / math.sqrt(in_features), generator=generator)
            self.lora_a = nn.Parameter(a)
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=weight.device, dtype=weight.dtype))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        if method in {"dora", "magnitude"}:
            magnitude = torch.linalg.vector_norm(weight, dim=1)
            self.magnitude = nn.Parameter(magnitude.detach().clone())
        else:
            self.register_parameter("magnitude", None)

    def effective_weight(self) -> torch.Tensor:
        if self.method == "frozen":
            return self.base_weight

        if self.method == "magnitude":
            direction_norm = torch.linalg.vector_norm(self.base_weight, dim=1, keepdim=True).clamp_min(1e-12)
            return self.base_weight * (self.magnitude[:, None] / direction_norm)

        delta = (self.lora_b @ self.lora_a) * self.scaling
        direction = self.base_weight + delta
        if self.method == "lora":
            return direction

        direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).detach().clamp_min(1e-12)
        return direction * (self.magnitude[:, None] / direction_norm)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.effective_weight(), self.base_bias)


class BaseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(inputs))
        hidden = F.gelu(self.fc2(hidden))
        return self.fc3(hidden)


class AdaptedMLP(nn.Module):
    def __init__(self, base: BaseMLP, *, method: AdapterMethod, rank: int, init_seed: int) -> None:
        super().__init__()
        self.fc1 = AdapterLinear(
            base.fc1.weight,
            base.fc1.bias,
            method=method,
            rank=rank,
            init_seed=init_seed + 11,
        )
        self.fc2 = AdapterLinear(
            base.fc2.weight,
            base.fc2.bias,
            method=method,
            rank=rank,
            init_seed=init_seed + 23,
        )
        self.fc3 = AdapterLinear(
            base.fc3.weight,
            base.fc3.bias,
            method=method,
            rank=rank,
            init_seed=init_seed + 37,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(inputs))
        hidden = F.gelu(self.fc2(hidden))
        return self.fc3(hidden)


def clone_for_full_finetuning(base: BaseMLP) -> BaseMLP:
    model = copy.deepcopy(base)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
