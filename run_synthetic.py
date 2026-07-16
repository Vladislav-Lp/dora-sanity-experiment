from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dora_study.synthetic import (  # noqa: E402
    constructed_dora,
    fit_dora,
    magnitude_oracle,
    make_problem,
    metric_record,
    svd_lora_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled DoRA/LoRA geometry sweep")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test sweep")
    parser.add_argument(
        "--with-optimization",
        action="store_true",
        help="Also fit DoRA with Adam; the default sweep isolates representational capacity.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "synthetic")
    return parser.parse_args()


def parameter_count(method: str, rank: int, in_features: int = 32, out_features: int = 16) -> int:
    if method in {"DoRA construction", "DoRA optimized"}:
        return rank * (in_features + out_features) + out_features
    if method == "LoRA SVD oracle":
        return rank * (in_features + out_features)
    if method == "Magnitude-only oracle":
        return out_features
    if method == "Full fine-tuning":
        return in_features * out_features
    return 0


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        seeds = [100, 101]
        ranks = [2, 4]
        magnitude_strengths = [0.0, 0.6]
        steps = 500
        learning_rates = (0.03, 0.1)
    else:
        seeds = list(range(100, 110))
        ranks = [1, 2, 4, 8]
        magnitude_strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
        steps = 1600
        learning_rates = (0.01, 0.03, 0.1)

    records: list[dict[str, object]] = []
    for seed in seeds:
        for magnitude_strength in magnitude_strengths:
            problem = make_problem(seed, magnitude_strength=magnitude_strength)
            baselines = {
                "Frozen": problem.base_weight,
                "Magnitude-only oracle": magnitude_oracle(problem.base_weight, problem.target_weight),
                "Full fine-tuning": problem.target_weight,
            }
            for rank in ranks:
                weights = dict(baselines)
                weights["LoRA SVD oracle"] = svd_lora_oracle(problem.base_weight, problem.target_weight, rank)
                weights["DoRA construction"] = constructed_dora(problem, rank)
                optimization_loss = np.nan
                selected_lr = np.nan
                selected_step = 0
                if args.with_optimization:
                    dora_weight, optimization_loss, selected_lr, selected_step = fit_dora(
                        problem,
                        rank=rank,
                        init_seed=10_000 + 101 * seed + rank,
                        steps=steps,
                        learning_rates=learning_rates,
                    )
                    weights["DoRA optimized"] = dora_weight

                for method, weight in weights.items():
                    record: dict[str, object] = {
                        "seed": seed,
                        "rank": rank,
                        "magnitude_strength": magnitude_strength,
                        "direction_strength": 0.35,
                        "true_direction_rank": 4,
                        "method": method,
                        "trainable_parameters": parameter_count(method, rank),
                        "selected_lr": selected_lr if method == "DoRA optimized" else np.nan,
                        "optimization_steps": selected_step if method == "DoRA optimized" else 0,
                        "optimization_loss": optimization_loss if method == "DoRA optimized" else np.nan,
                    }
                    record.update(metric_record(weight, problem.target_weight))
                    records.append(record)

    results = pd.DataFrame(records)
    results.to_csv(args.output_dir / "synthetic_runs.csv", index=False)

    summary = (
        results.groupby(["method", "rank", "magnitude_strength", "trainable_parameters"], as_index=False)
        .agg(
            relative_weight_error_mean=("relative_weight_error", "mean"),
            relative_weight_error_std=("relative_weight_error", "std"),
            expected_output_mse_mean=("expected_output_mse", "mean"),
            expected_output_mse_std=("expected_output_mse", "std"),
            direction_error_mean=("direction_error", "mean"),
            magnitude_relative_mae_mean=("magnitude_relative_mae", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(args.output_dir / "synthetic_summary.csv", index=False)

    paired = results.pivot_table(
        index=["seed", "rank", "magnitude_strength"],
        columns="method",
        values="relative_weight_error",
    ).reset_index()
    paired["dora_vs_lora_oracle_ratio"] = paired["DoRA construction"] / paired[
        "LoRA SVD oracle"
    ].clip(lower=1e-12)
    paired.to_csv(args.output_dir / "synthetic_paired.csv", index=False)

    metadata = {
        "seeds": seeds,
        "ranks": ranks,
        "magnitude_strengths": magnitude_strengths,
        "steps": steps,
        "learning_rates": learning_rates,
        "torch_version": torch.__version__,
        "quick": args.quick,
        "with_optimization": args.with_optimization,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
