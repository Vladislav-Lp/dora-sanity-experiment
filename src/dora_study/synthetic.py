from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .models import AdapterLinear


@dataclass(frozen=True)
class SyntheticProblem:
    base_weight: torch.Tensor
    target_weight: torch.Tensor
    direction_delta: torch.Tensor
    row_multipliers: torch.Tensor


def make_problem(
    seed: int,
    *,
    in_features: int = 32,
    out_features: int = 16,
    true_rank: int = 4,
    direction_strength: float = 0.35,
    magnitude_strength: float = 0.0,
) -> SyntheticProblem:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(out_features, in_features, generator=generator) / np.sqrt(in_features)

    left = torch.randn(out_features, true_rank, generator=generator)
    right = torch.randn(true_rank, in_features, generator=generator)
    delta = left @ right
    delta = delta * (direction_strength * torch.linalg.vector_norm(base) / torch.linalg.vector_norm(delta))

    row_axis = torch.linspace(-1.0, 1.0, out_features)
    permutation = torch.randperm(out_features, generator=generator)
    row_axis = row_axis[permutation]
    multipliers = 1.0 + magnitude_strength * row_axis

    target = (base + delta) * multipliers[:, None]
    return SyntheticProblem(base, target, delta, multipliers)


def relative_weight_error(weight: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(weight - target) / torch.linalg.vector_norm(target))


def output_mse(weight: torch.Tensor, target: torch.Tensor) -> float:
    # Expected per-output MSE for x ~ N(0, I).
    return float(torch.sum((weight - target) ** 2) / target.shape[0])


def direction_error(weight: torch.Tensor, target: torch.Tensor) -> float:
    weight_unit = weight / torch.linalg.vector_norm(weight, dim=1, keepdim=True).clamp_min(1e-12)
    target_unit = target / torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(1e-12)
    return float(torch.mean(1.0 - torch.sum(weight_unit * target_unit, dim=1)))


def magnitude_relative_mae(weight: torch.Tensor, target: torch.Tensor) -> float:
    weight_norm = torch.linalg.vector_norm(weight, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12)
    return float(torch.mean(torch.abs(weight_norm / target_norm - 1.0)))


def svd_lora_oracle(base: torch.Tensor, target: torch.Tensor, rank: int) -> torch.Tensor:
    delta = target - base
    left, singular_values, right_t = torch.linalg.svd(delta, full_matrices=False)
    approximation = (left[:, :rank] * singular_values[:rank]) @ right_t[:rank]
    return base + approximation


def magnitude_oracle(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    base_norm = torch.linalg.vector_norm(base, dim=1, keepdim=True).clamp_min(1e-12)
    target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
    return base * (target_norm / base_norm)


def constructed_dora(problem: SyntheticProblem, rank: int) -> torch.Tensor:
    """A feasible rank-r DoRA solution using the known synthetic decomposition.

    This is a representational construction, not a learned estimator. It makes
    the capacity comparison conservative and reproducible: LoRA receives its
    exact SVD optimum, while DoRA receives a valid (not necessarily optimal for
    rank < true_rank) parameterization.
    """
    left, singular_values, right_t = torch.linalg.svd(problem.direction_delta, full_matrices=False)
    delta_rank = (left[:, :rank] * singular_values[:rank]) @ right_t[:rank]
    direction = problem.base_weight + delta_rank
    direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12)
    target_norm = torch.linalg.vector_norm(problem.target_weight, dim=1, keepdim=True)
    return direction * (target_norm / direction_norm)


def fit_dora(
    problem: SyntheticProblem,
    *,
    rank: int,
    init_seed: int,
    steps: int,
    learning_rates: tuple[float, ...],
) -> tuple[torch.Tensor, float, float, int]:
    best_weight: torch.Tensor | None = None
    best_loss = float("inf")
    best_lr = float("nan")
    best_step = 0

    for lr in learning_rates:
        model = AdapterLinear(
            problem.base_weight,
            method="dora",
            rank=rank,
            init_seed=init_seed,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        local_best = float("inf")
        stale_steps = 0

        for step in range(1, steps + 1):
            estimate = model.effective_weight()
            loss = torch.mean((estimate - problem.target_weight) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value = float(loss.detach())
            if value < local_best * (1.0 - 1e-7):
                local_best = value
                stale_steps = 0
            else:
                stale_steps += 1
            if value < 1e-13 or stale_steps >= 250:
                break

        with torch.no_grad():
            final_weight = model.effective_weight().detach().clone()
            final_loss = float(torch.mean((final_weight - problem.target_weight) ** 2))
        if final_loss < best_loss:
            best_weight = final_weight
            best_loss = final_loss
            best_lr = lr
            best_step = step

    if best_weight is None:
        raise RuntimeError("DoRA optimization did not produce a result")
    return best_weight, best_loss, best_lr, best_step


def metric_record(weight: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "relative_weight_error": relative_weight_error(weight, target),
        "expected_output_mse": output_mse(weight, target),
        "direction_error": direction_error(weight, target),
        "magnitude_relative_mae": magnitude_relative_mae(weight, target),
    }
