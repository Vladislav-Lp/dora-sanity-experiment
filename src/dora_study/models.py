from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F


AdapterMethod = Literal["frozen", "magnitude", "lora", "dora"]
RankSpec: TypeAlias = int | Sequence[int]


def _rank_allocation(rank: RankSpec, n_layers: int) -> tuple[int, ...]:
    if isinstance(rank, int):
        allocation = (rank,) * n_layers
    else:
        allocation = tuple(int(value) for value in rank)
    if len(allocation) != n_layers:
        raise ValueError(f"expected {n_layers} ranks, received {len(allocation)}")
    if any(value < 0 for value in allocation):
        raise ValueError("ranks cannot be negative")
    return allocation


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


class AdapterConv2d(nn.Module):
    """Frozen 2D convolution with a matrix-shaped LoRA/DoRA update.

    Each output filter is treated as one row of a matrix. This is the same
    decomposition used by common PEFT implementations for convolutional
    weights and lets the DoRA magnitude remain one scalar per output channel.
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
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if weight.ndim != 4:
            raise ValueError("weight must be a 4D convolution tensor")
        if method in {"lora", "dora"} and rank < 1:
            raise ValueError("rank must be positive for LoRA and DoRA")

        self.method = method
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer("base_weight", weight.detach().clone())
        if bias is None:
            self.register_buffer("base_bias", None)
        else:
            self.register_buffer("base_bias", bias.detach().clone())

        out_channels = weight.shape[0]
        flattened_inputs = int(weight[0].numel())
        self.scaling = float(alpha if alpha is not None else rank) / max(rank, 1)

        if method in {"lora", "dora"}:
            generator = torch.Generator(device=weight.device).manual_seed(init_seed)
            a = torch.empty(rank, flattened_inputs, device=weight.device, dtype=weight.dtype)
            a.uniform_(
                -1.0 / math.sqrt(flattened_inputs),
                1.0 / math.sqrt(flattened_inputs),
                generator=generator,
            )
            self.lora_a = nn.Parameter(a)
            self.lora_b = nn.Parameter(
                torch.zeros(out_channels, rank, device=weight.device, dtype=weight.dtype)
            )
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        if method in {"dora", "magnitude"}:
            magnitude = torch.linalg.vector_norm(weight.flatten(1), dim=1)
            self.magnitude = nn.Parameter(magnitude.detach().clone())
        else:
            self.register_parameter("magnitude", None)

    def effective_weight(self) -> torch.Tensor:
        if self.method == "frozen":
            return self.base_weight

        flat_base = self.base_weight.flatten(1)
        if self.method == "magnitude":
            direction_norm = torch.linalg.vector_norm(flat_base, dim=1, keepdim=True).clamp_min(1e-12)
            return (flat_base * (self.magnitude[:, None] / direction_norm)).view_as(self.base_weight)

        delta = (self.lora_b @ self.lora_a) * self.scaling
        direction = flat_base + delta
        if self.method == "lora":
            return direction.view_as(self.base_weight)

        direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).detach().clamp_min(1e-12)
        return (direction * (self.magnitude[:, None] / direction_norm)).view_as(self.base_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            self.effective_weight(),
            self.base_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


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
    def __init__(self, base: BaseMLP, *, method: AdapterMethod, rank: RankSpec, init_seed: int) -> None:
        super().__init__()
        self.rank_allocation = _rank_allocation(rank, 3)
        self.fc1 = AdapterLinear(
            base.fc1.weight,
            base.fc1.bias,
            method=method,
            rank=self.rank_allocation[0],
            init_seed=init_seed + 11,
        )
        self.fc2 = AdapterLinear(
            base.fc2.weight,
            base.fc2.bias,
            method=method,
            rank=self.rank_allocation[1],
            init_seed=init_seed + 23,
        )
        self.fc3 = AdapterLinear(
            base.fc3.weight,
            base.fc3.bias,
            method=method,
            rank=self.rank_allocation[2],
            init_seed=init_seed + 37,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(inputs))
        hidden = F.gelu(self.fc2(hidden))
        return self.fc3(hidden)


class BaseCNN(nn.Module):
    """Compact second-backbone check for 8x8 Digits images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        images = inputs.reshape(-1, 1, 8, 8)
        hidden = F.max_pool2d(F.gelu(self.conv1(images)), kernel_size=2)
        hidden = F.max_pool2d(F.gelu(self.conv2(hidden)), kernel_size=2)
        hidden = hidden.flatten(1)
        hidden = F.gelu(self.fc1(hidden))
        return self.fc2(hidden)


class AdaptedCNN(nn.Module):
    def __init__(self, base: BaseCNN, *, method: AdapterMethod, rank: RankSpec, init_seed: int) -> None:
        super().__init__()
        self.rank_allocation = _rank_allocation(rank, 4)
        self.conv1 = AdapterConv2d(
            base.conv1.weight,
            base.conv1.bias,
            method=method,
            rank=self.rank_allocation[0],
            init_seed=init_seed + 11,
            stride=base.conv1.stride,
            padding=base.conv1.padding,
            dilation=base.conv1.dilation,
            groups=base.conv1.groups,
        )
        self.conv2 = AdapterConv2d(
            base.conv2.weight,
            base.conv2.bias,
            method=method,
            rank=self.rank_allocation[1],
            init_seed=init_seed + 23,
            stride=base.conv2.stride,
            padding=base.conv2.padding,
            dilation=base.conv2.dilation,
            groups=base.conv2.groups,
        )
        self.fc1 = AdapterLinear(
            base.fc1.weight,
            base.fc1.bias,
            method=method,
            rank=self.rank_allocation[2],
            init_seed=init_seed + 37,
        )
        self.fc2 = AdapterLinear(
            base.fc2.weight,
            base.fc2.bias,
            method=method,
            rank=self.rank_allocation[3],
            init_seed=init_seed + 53,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        images = inputs.reshape(-1, 1, 8, 8)
        hidden = F.max_pool2d(F.gelu(self.conv1(images)), kernel_size=2)
        hidden = F.max_pool2d(F.gelu(self.conv2(hidden)), kernel_size=2)
        hidden = hidden.flatten(1)
        hidden = F.gelu(self.fc1(hidden))
        return self.fc2(hidden)


def clone_for_full_finetuning(base: nn.Module) -> nn.Module:
    model = copy.deepcopy(base)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
