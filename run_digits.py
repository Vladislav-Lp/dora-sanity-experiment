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

from dora_study.digits import (  # noqa: E402
    balanced_subset,
    corrupt,
    evaluate,
    load_split,
    pretrain_base,
    train_candidate,
)
from dora_study.models import count_trainable_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot domain adaptation on sklearn Digits")
    parser.add_argument("--quick", action="store_true", help="Run one seed, one scenario, and two ranks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "digits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split()
    base = pretrain_base(split)
    clean_metrics = evaluate(base, split.x_test, split.y_test)

    if args.quick:
        seeds = [11]
        scenarios = ["mixed"]
        ranks = [2, 4]
        adapter_lrs = [0.01, 0.03]
        magnitude_lrs = [0.01, 0.03]
        full_lrs = [0.0003, 0.001]
        max_epochs = 60
        patience = 10
    else:
        seeds = [11, 22, 33, 44, 55]
        scenarios = ["contrast", "rotation", "mixed"]
        ranks = [1, 2, 4, 8]
        adapter_lrs = [0.003, 0.01, 0.03]
        magnitude_lrs = [0.003, 0.01, 0.03]
        full_lrs = [0.0001, 0.0003, 0.001]
        max_epochs = 120
        patience = 18

    records: list[dict[str, object]] = []
    fixed_corruptions = {
        scenario: {
            "val": corrupt(split.x_val, scenario, seed=90_000 + i),
            "test": corrupt(split.x_test, scenario, seed=91_000 + i),
        }
        for i, scenario in enumerate(scenarios)
    }

    started = time.time()
    for scenario_index, scenario in enumerate(scenarios):
        x_val = fixed_corruptions[scenario]["val"]
        x_test = fixed_corruptions[scenario]["test"]
        frozen_shift_metrics = evaluate(base, x_test, split.y_test)

        for seed in seeds:
            clean_subset, y_subset = balanced_subset(split.x_train, split.y_train, per_class=40, seed=seed)
            x_subset = corrupt(clean_subset, scenario, seed=70_000 + 100 * scenario_index + seed)

            frozen_record: dict[str, object] = {
                "scenario": scenario,
                "seed": seed,
                "method": "Frozen",
                "rank": 0,
                "selected_learning_rate": 0.0,
                "validation_accuracy": evaluate(base, x_val, split.y_val)["accuracy"],
                "validation_nll": evaluate(base, x_val, split.y_val)["nll"],
                "best_epoch": 0,
                "epochs_ran": 0,
                "trainable_parameters": 0,
                "target_accuracy": frozen_shift_metrics["accuracy"],
                "target_macro_f1": frozen_shift_metrics["macro_f1"],
                "target_nll": frozen_shift_metrics["nll"],
                "clean_accuracy": clean_metrics["accuracy"],
                "clean_macro_f1": clean_metrics["macro_f1"],
            }
            records.append(frozen_record)

            configurations = [("magnitude", 0, magnitude_lrs), ("full", 0, full_lrs)]
            configurations.extend((method, rank, adapter_lrs) for method in ["lora", "dora"] for rank in ranks)

            for method, rank, learning_rates in configurations:
                candidates: list[tuple[tuple[float, float], object, dict[str, float | int], float]] = []
                for learning_rate in learning_rates:
                    candidate, training_info = train_candidate(
                        base,
                        method=method,
                        rank=rank,
                        learning_rate=learning_rate,
                        x_train=x_subset,
                        y_train=y_subset,
                        x_val=x_val,
                        y_val=split.y_val,
                        seed=100_000 + 1_000 * scenario_index + 10 * seed + rank,
                        max_epochs=max_epochs,
                        patience=patience,
                    )
                    score = (float(training_info["validation_accuracy"]), -float(training_info["validation_nll"]))
                    candidates.append((score, candidate, training_info, learning_rate))

                _, best_model, training_info, learning_rate = max(candidates, key=lambda item: item[0])
                target_metrics = evaluate(best_model, x_test, split.y_test)
                retained_metrics = evaluate(best_model, split.x_test, split.y_test)
                display_method = {
                    "lora": "LoRA",
                    "dora": "DoRA",
                    "magnitude": "Magnitude-only",
                    "full": "Full fine-tuning",
                }[method]
                records.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": display_method,
                        "rank": rank,
                        "selected_learning_rate": learning_rate,
                        **training_info,
                        "target_accuracy": target_metrics["accuracy"],
                        "target_macro_f1": target_metrics["macro_f1"],
                        "target_nll": target_metrics["nll"],
                        "clean_accuracy": retained_metrics["accuracy"],
                        "clean_macro_f1": retained_metrics["macro_f1"],
                    }
                )

            print(f"completed scenario={scenario} seed={seed}", flush=True)

    results = pd.DataFrame(records)
    results.to_csv(args.output_dir / "digits_runs.csv", index=False)
    summary = (
        results.groupby(["scenario", "method", "rank", "trainable_parameters"], as_index=False)
        .agg(
            target_accuracy_mean=("target_accuracy", "mean"),
            target_accuracy_std=("target_accuracy", "std"),
            target_macro_f1_mean=("target_macro_f1", "mean"),
            clean_accuracy_mean=("clean_accuracy", "mean"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(args.output_dir / "digits_summary.csv", index=False)
    metadata = {
        "dataset": "sklearn.datasets.load_digits",
        "split_seed": 2026,
        "adaptation_examples": 400,
        "seeds": seeds,
        "scenarios": scenarios,
        "ranks": ranks,
        "adapter_learning_rates": adapter_lrs,
        "magnitude_learning_rates": magnitude_lrs,
        "full_finetuning_learning_rates": full_lrs,
        "base_clean_test_metrics": clean_metrics,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "quick": args.quick,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
