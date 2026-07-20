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
    balanced_nested_indices,
    corrupt,
    evaluate,
    load_split,
    pretrain_base,
    train_candidate,
)


DATA_SEEDS = tuple(range(301, 321))
PER_CLASS_VALUES = (5, 10, 20, 40)
METHOD_IDS = ("lora", "lora_plus", "dora", "lora_matched", "dora_budgeted")
IMPLEMENTATION = {
    "lora": "lora",
    "lora_plus": "lora_plus",
    "dora": "dora",
    "lora_matched": "lora",
    "dora_budgeted": "dora",
}
LABELS = {
    "lora": "LoRA",
    "lora_plus": "LoRA+",
    "dora": "DoRA",
    "lora_matched": "LoRA (DoRA-matched budget)",
    "dora_budgeted": "DoRA (LoRA-budget ceiling)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preregistered mixed-shift target-data sweep")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "data_sweep_mlp")
    return parser.parse_args()


def parse_allocation(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


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

    selection_path = ROOT / "results" / "confirmatory_mlp" / "selected_configs.csv"
    selected = pd.read_csv(selection_path, dtype={"allocation": str})
    selected = selected[(selected["scenario"] == "mixed") & selected["method_id"].isin(METHOD_IDS)]
    if set(selected["method_id"]) != set(METHOD_IDS) or selected["method_id"].duplicated().any():
        raise RuntimeError("mixed-shift MLP selection is incomplete or non-unique")
    configs = selected.set_index("method_id").to_dict(orient="index")

    seeds = DATA_SEEDS[:2] if args.quick else DATA_SEEDS
    per_class_values = (5, 20) if args.quick else PER_CLASS_VALUES
    max_epochs = 25 if args.quick else 120
    patience = 5 if args.quick else 18
    split = load_split()
    base = pretrain_base(split, seed=2026, epochs=40 if args.quick else 160, architecture="mlp")
    clean_metrics = evaluate(base, split.x_test, split.y_test)
    x_val = corrupt(split.x_val, "mixed", seed=90_002)
    x_test = corrupt(split.x_test, "mixed", seed=91_002)

    records: list[dict[str, object]] = []
    started = time.time()
    for seed in seeds:
        shifted_train = corrupt(split.x_train, "mixed", seed=170_000 + seed)
        for per_class in per_class_values:
            indices = balanced_nested_indices(split.y_train, per_class=per_class, seed=seed)
            x_train = shifted_train[indices]
            y_train = split.y_train[indices]
            for method_id in METHOD_IDS:
                config = configs[method_id]
                implementation = IMPLEMENTATION[method_id]
                model, training_info = train_candidate(
                    base,
                    method=implementation,
                    rank=parse_allocation(str(config["allocation"])),
                    learning_rate=float(config["learning_rate"]),
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=split.y_val,
                    seed=800_000 + seed,
                    max_epochs=max_epochs,
                    patience=patience,
                    lora_plus_ratio=16.0,
                )
                target_metrics = evaluate(model, x_test, split.y_test)
                retained_metrics = evaluate(model, split.x_test, split.y_test)
                records.append(
                    {
                        "architecture": "mlp",
                        "scenario": "mixed",
                        "seed": seed,
                        "per_class": per_class,
                        "adaptation_examples": 10 * per_class,
                        "method_id": method_id,
                        "method": LABELS[method_id],
                        "implementation": implementation,
                        "allocation": str(config["allocation"]),
                        "selected_learning_rate": float(config["learning_rate"]),
                        "lora_plus_ratio": 16.0 if method_id == "lora_plus" else 1.0,
                        **training_info,
                        "target_accuracy": target_metrics["accuracy"],
                        "target_macro_f1": target_metrics["macro_f1"],
                        "target_nll": target_metrics["nll"],
                        "clean_accuracy": retained_metrics["accuracy"],
                        "clean_macro_f1": retained_metrics["macro_f1"],
                        "clean_nll": retained_metrics["nll"],
                    }
                )
                checkpoint(records, args.output_dir / "data_sweep_runs.csv")
            print(f"data-sweep seed={seed} per_class={per_class}", flush=True)

    runs = pd.DataFrame(records)
    summary = (
        runs.groupby(
            [
                "per_class",
                "adaptation_examples",
                "method_id",
                "method",
                "allocation",
                "selected_learning_rate",
                "trainable_parameters",
            ],
            as_index=False,
        )
        .agg(
            target_accuracy_mean=("target_accuracy", "mean"),
            target_accuracy_std=("target_accuracy", "std"),
            target_macro_f1_mean=("target_macro_f1", "mean"),
            target_nll_mean=("target_nll", "mean"),
            clean_accuracy_mean=("clean_accuracy", "mean"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            train_seconds_sum=("train_seconds", "sum"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(args.output_dir / "data_sweep_summary.csv", index=False)
    metadata = {
        "protocol": "docs/EXTENSION_PROTOCOL.md",
        "selection_source": str(selection_path.relative_to(ROOT)),
        "scenario": "mixed",
        "seeds": seeds,
        "per_class_values": per_class_values,
        "adaptation_examples": [10 * value for value in per_class_values],
        "nested_subsets": True,
        "same_corrupted_sample_realization_across_budgets": True,
        "base_clean_test_metrics": clean_metrics,
        "max_epochs": max_epochs,
        "patience": patience,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "quick": args.quick,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
