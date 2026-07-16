from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

from .models import AdaptedMLP, BaseMLP, clone_for_full_finetuning, count_trainable_parameters


@dataclass(frozen=True)
class DigitsSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(seed: int = 2026) -> DigitsSplit:
    features, labels = load_digits(return_X_y=True)
    features = (features.astype(np.float32) / 16.0).clip(0.0, 1.0)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed + 1,
        stratify=y_train_val,
    )
    return DigitsSplit(x_train, y_train, x_val, y_val, x_test, y_test)


def corrupt(features: np.ndarray, scenario: str, *, seed: int) -> np.ndarray:
    images = features.reshape(-1, 8, 8).copy()
    rng = np.random.default_rng(seed)

    if scenario == "contrast":
        transformed = images * 0.10
    elif scenario == "rotation":
        transformed = np.stack(
            [rotate(image, angle=25.0, reshape=False, order=1, mode="constant", cval=0.0) for image in images]
        )
    elif scenario == "mixed":
        transformed = np.stack(
            [rotate(image, angle=18.0, reshape=False, order=1, mode="constant", cval=0.0) for image in images]
        )
        transformed = transformed * 0.65
        transformed = transformed + rng.normal(0.0, 0.16, size=transformed.shape)
    else:
        raise ValueError(f"unknown corruption scenario: {scenario}")

    return np.clip(transformed, 0.0, 1.0).reshape(-1, 64).astype(np.float32)


def balanced_subset(features: np.ndarray, labels: np.ndarray, *, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError("not enough examples for the requested balanced subset")
        indices.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    rng.shuffle(indices)
    return features[indices], labels[indices]


def _tensor(array: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = torch.from_numpy(array)
    return value.to(dtype=dtype) if dtype is not None else value


def _batches(n_samples: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    order = torch.randperm(n_samples, generator=generator)
    return [order[start : start + batch_size] for start in range(0, n_samples, batch_size)]


def evaluate(model: nn.Module, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(_tensor(features, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(10))),
    }


def pretrain_base(split: DigitsSplit, *, seed: int = 2026, epochs: int = 160) -> BaseMLP:
    seed_everything(seed)
    model = BaseMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    x_train = _tensor(split.x_train, dtype=torch.float32)
    y_train = _tensor(split.y_train, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed + 99)

    for _ in range(epochs):
        model.train()
        for batch in _batches(len(x_train), 64, generator):
            loss = F.cross_entropy(model(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def build_candidate(base: BaseMLP, *, method: str, rank: int, seed: int) -> nn.Module:
    if method == "full":
        return clone_for_full_finetuning(base)
    if method in {"lora", "dora", "magnitude"}:
        return AdaptedMLP(base, method=method, rank=rank, init_seed=seed)
    raise ValueError(f"unsupported method: {method}")


def train_candidate(
    base: BaseMLP,
    *,
    method: str,
    rank: int,
    learning_rate: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    max_epochs: int = 120,
    patience: int = 18,
) -> tuple[nn.Module, dict[str, float | int]]:
    seed_everything(seed)
    model = build_candidate(base, method=method, rank=rank, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x_train_t = _tensor(x_train, dtype=torch.float32)
    y_train_t = _tensor(y_train, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed + 13)

    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in _batches(len(x_train_t), 64, generator):
            loss = F.cross_entropy(model(x_train_t[batch]), y_train_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, x_val, y_val)
        better = (val_metrics["accuracy"] > best_accuracy + 1e-12) or (
            abs(val_metrics["accuracy"] - best_accuracy) <= 1e-12 and val_metrics["nll"] < best_loss
        )
        if better:
            best_accuracy = val_metrics["accuracy"]
            best_loss = val_metrics["nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "validation_accuracy": best_accuracy,
        "validation_nll": best_loss,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "trainable_parameters": count_trainable_parameters(model),
    }
