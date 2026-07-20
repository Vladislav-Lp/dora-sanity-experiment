from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dora_study.synthetic import (  # noqa: E402
    constructed_dora,
    fit_dora,
    make_problem,
    metric_record,
    svd_lora_oracle,
)


PROBLEM_SEEDS = tuple(range(200, 210))
INITIALIZATIONS = tuple(range(5))
MAGNITUDE_STRENGTHS = (0.0, 0.4, 0.8)
LEARNING_RATES = (0.01, 0.03, 0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DoRA synthetic optimization diagnostic")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "synthetic_optimization",
    )
    return parser.parse_args()


def checkpoint(records: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")

    problem_seeds = PROBLEM_SEEDS[:2] if args.quick else PROBLEM_SEEDS
    initializations = INITIALIZATIONS[:2] if args.quick else INITIALIZATIONS
    strengths = (0.0, 0.8) if args.quick else MAGNITUDE_STRENGTHS
    steps = 400 if args.quick else 2_000
    rank = 4

    records: list[dict[str, object]] = []
    started = time.time()
    for problem_seed in problem_seeds:
        for magnitude_strength in strengths:
            problem = make_problem(problem_seed, magnitude_strength=magnitude_strength)
            lora_oracle = svd_lora_oracle(problem.base_weight, problem.target_weight, rank=rank)
            feasible_dora = constructed_dora(problem, rank=rank)
            lora_metrics = metric_record(lora_oracle, problem.target_weight)
            feasible_metrics = metric_record(feasible_dora, problem.target_weight)
            for initialization in initializations:
                init_seed = 1_000_000 + 101 * problem_seed + initialization
                optimized, loss, selected_lr, selected_step = fit_dora(
                    problem,
                    rank=rank,
                    init_seed=init_seed,
                    steps=steps,
                    learning_rates=LEARNING_RATES,
                )
                optimized_metrics = metric_record(optimized, problem.target_weight)
                records.append(
                    {
                        "problem_seed": problem_seed,
                        "initialization": initialization,
                        "init_seed": init_seed,
                        "rank": rank,
                        "magnitude_strength": magnitude_strength,
                        "true_direction_rank": 4,
                        "direction_strength": 0.35,
                        "selected_learning_rate": selected_lr,
                        "optimization_steps": selected_step,
                        "optimization_mse": loss,
                        **{f"optimized_dora_{key}": value for key, value in optimized_metrics.items()},
                        **{f"feasible_dora_{key}": value for key, value in feasible_metrics.items()},
                        **{f"lora_oracle_{key}": value for key, value in lora_metrics.items()},
                    }
                )
                checkpoint(records, args.output_dir / "synthetic_optimization_runs.csv")
            print(
                f"synthetic-optimization problem={problem_seed} gamma={magnitude_strength:g}",
                flush=True,
            )

    runs = pd.DataFrame(records)
    metric_columns = [
        "optimized_dora_relative_weight_error",
        "optimized_dora_expected_output_mse",
        "optimized_dora_direction_error",
        "optimized_dora_magnitude_relative_mae",
        "feasible_dora_relative_weight_error",
        "lora_oracle_relative_weight_error",
    ]
    problem_summary = (
        runs.groupby(["problem_seed", "magnitude_strength"], as_index=False)
        .agg(
            **{f"{column}_mean": (column, "mean") for column in metric_columns},
            optimized_dora_relative_weight_error_median=(
                "optimized_dora_relative_weight_error",
                "median",
            ),
            optimized_dora_relative_weight_error_max=(
                "optimized_dora_relative_weight_error",
                "max",
            ),
            convergence_rate_lt_1e3=(
                "optimized_dora_relative_weight_error",
                lambda values: float((values < 1e-3).mean()),
            ),
            n_initializations=("initialization", "nunique"),
        )
    )
    problem_summary.to_csv(args.output_dir / "synthetic_optimization_problem_summary.csv", index=False)
    summary = (
        problem_summary.groupby("magnitude_strength", as_index=False)
        .agg(
            optimized_error_mean=("optimized_dora_relative_weight_error_mean", "mean"),
            optimized_error_std_across_problems=("optimized_dora_relative_weight_error_mean", "std"),
            optimized_worst_init_error_mean=("optimized_dora_relative_weight_error_max", "mean"),
            feasible_dora_error_mean=("feasible_dora_relative_weight_error_mean", "mean"),
            lora_oracle_error_mean=("lora_oracle_relative_weight_error_mean", "mean"),
            convergence_rate_mean=("convergence_rate_lt_1e3", "mean"),
            n_problems=("problem_seed", "nunique"),
        )
    )
    summary.to_csv(args.output_dir / "synthetic_optimization_summary.csv", index=False)
    metadata = {
        "protocol": "docs/EXTENSION_PROTOCOL.md",
        "problem_seeds": problem_seeds,
        "initializations": initializations,
        "magnitude_strengths": strengths,
        "rank": rank,
        "steps": steps,
        "learning_rates": LEARNING_RATES,
        "aggregation_unit": "problem_seed",
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "quick": args.quick,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
