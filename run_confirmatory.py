from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
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


SCENARIOS = ("contrast", "rotation", "mixed")
PILOT_SEEDS = (11, 22, 33, 44, 55)
MLP_CONFIRMATORY_SEEDS = tuple(range(101, 121))
CNN_CONFIRMATORY_SEEDS = tuple(range(201, 211))
STANDARD_LRS = (0.003, 0.01, 0.03)
FULL_LRS = (0.0001, 0.0003, 0.001)
LORA_PLUS_LRS = (0.0003, 0.001, 0.003)
LORA_PLUS_RATIO = 16.0


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    implementation: str
    allocations: tuple[tuple[int, ...], ...]
    learning_rates: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preregistered validation-selection and confirmatory Digits adaptation study"
    )
    parser.add_argument("--architecture", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--quick", action="store_true", help="Small end-to-end smoke run")
    parser.add_argument("--pilot-only", action="store_true", help="Stop before any test-set evaluation")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def uniform(rank: int, architecture: str) -> tuple[int, ...]:
    return (rank,) * (3 if architecture == "mlp" else 4)


def method_specs(architecture: str, *, quick: bool) -> tuple[MethodSpec, ...]:
    standard_lrs = STANDARD_LRS[:2] if quick else STANDARD_LRS
    full_lrs = FULL_LRS[:2] if quick else FULL_LRS
    lora_plus_lrs = LORA_PLUS_LRS[:2] if quick else LORA_PLUS_LRS
    zero = uniform(0, architecture)
    rank4 = uniform(4, architecture)
    specs = [
        MethodSpec("frozen", "Frozen", "frozen", (zero,), (0.0,)),
        MethodSpec("magnitude", "Magnitude-only", "magnitude", (zero,), standard_lrs),
        MethodSpec("lora", "LoRA", "lora", (rank4,), standard_lrs),
        MethodSpec("lora_plus", "LoRA+", "lora_plus", (rank4,), lora_plus_lrs),
        MethodSpec("dora", "DoRA", "dora", (rank4,), standard_lrs),
        MethodSpec("full", "Full fine-tuning", "full", (zero,), full_lrs),
    ]
    if architecture == "mlp":
        specs.extend(
            [
                MethodSpec(
                    "lora_matched",
                    "LoRA (DoRA-matched budget)",
                    "lora",
                    ((5, 4, 4), (4, 5, 4), (4, 4, 6)),
                    standard_lrs,
                ),
                MethodSpec(
                    "dora_budgeted",
                    "DoRA (LoRA-budget ceiling)",
                    "dora",
                    ((4, 3, 3), (3, 4, 3), (3, 3, 6)),
                    standard_lrs,
                ),
            ]
        )
    return tuple(specs)


def allocation_text(allocation: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in allocation)


def parse_allocation(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def model_seed(architecture: str, scenario_index: int, adaptation_seed: int) -> int:
    architecture_offset = 0 if architecture == "mlp" else 500_000
    return 100_000 + architecture_offset + 1_000 * scenario_index + adaptation_seed


def target_data(split, scenario: str, scenario_index: int, seed: int, *, per_class: int = 40):
    clean_subset, labels = balanced_subset(
        split.x_train,
        split.y_train,
        per_class=per_class,
        seed=seed,
    )
    shifted_subset = corrupt(
        clean_subset,
        scenario,
        seed=70_000 + 100 * scenario_index + seed,
    )
    return shifted_subset, labels


def fixed_corruptions(split) -> dict[str, dict[str, object]]:
    return {
        scenario: {
            "val": corrupt(split.x_val, scenario, seed=90_000 + index),
            "test": corrupt(split.x_test, scenario, seed=91_000 + index),
        }
        for index, scenario in enumerate(SCENARIOS)
    }


def save_records(records: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def run_pilot(
    *,
    base: torch.nn.Module,
    split,
    corruptions: dict[str, dict[str, object]],
    architecture: str,
    specs: tuple[MethodSpec, ...],
    pilot_seeds: tuple[int, ...],
    max_epochs: int,
    patience: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        x_val = corruptions[scenario]["val"]
        frozen_validation = evaluate(base, x_val, split.y_val)
        for spec in specs:
            for allocation in spec.allocations:
                for learning_rate in spec.learning_rates:
                    for seed in pilot_seeds:
                        if spec.implementation == "frozen":
                            training_info: dict[str, float | int] = {
                                "validation_accuracy": frozen_validation["accuracy"],
                                "validation_nll": frozen_validation["nll"],
                                "best_epoch": 0,
                                "epochs_ran": 0,
                                "trainable_parameters": 0,
                                "train_seconds": 0.0,
                            }
                        else:
                            x_train, y_train = target_data(
                                split,
                                scenario,
                                scenario_index,
                                seed,
                            )
                            _, training_info = train_candidate(
                                base,
                                method=spec.implementation,
                                rank=allocation,
                                learning_rate=learning_rate,
                                x_train=x_train,
                                y_train=y_train,
                                x_val=x_val,
                                y_val=split.y_val,
                                seed=model_seed(architecture, scenario_index, seed),
                                max_epochs=max_epochs,
                                patience=patience,
                                lora_plus_ratio=LORA_PLUS_RATIO,
                            )
                        records.append(
                            {
                                "architecture": architecture,
                                "scenario": scenario,
                                "pilot_seed": seed,
                                "method_id": spec.method_id,
                                "method": spec.label,
                                "implementation": spec.implementation,
                                "allocation": allocation_text(allocation),
                                "learning_rate": learning_rate,
                                "lora_plus_ratio": LORA_PLUS_RATIO if spec.method_id == "lora_plus" else 1.0,
                                **training_info,
                            }
                        )
                    save_records(records, output_dir / "pilot_runs.csv")
                    print(
                        f"pilot architecture={architecture} scenario={scenario} method={spec.method_id} "
                        f"allocation={allocation_text(allocation)} lr={learning_rate:g}",
                        flush=True,
                    )

    runs = pd.DataFrame(records)
    summary = (
        runs.groupby(
            [
                "architecture",
                "scenario",
                "method_id",
                "method",
                "implementation",
                "allocation",
                "learning_rate",
                "lora_plus_ratio",
                "trainable_parameters",
            ],
            as_index=False,
        )
        .agg(
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_std=("validation_accuracy", "std"),
            validation_nll_mean=("validation_nll", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            train_seconds_sum=("train_seconds", "sum"),
            n_pilot_seeds=("pilot_seed", "nunique"),
        )
    )
    summary.to_csv(output_dir / "pilot_summary.csv", index=False)

    selected_rows: list[pd.Series] = []
    for _, group in summary.groupby(["architecture", "scenario", "method_id"], sort=False):
        ordered = group.sort_values(
            ["validation_accuracy_mean", "validation_nll_mean", "allocation", "learning_rate"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        selected_rows.append(ordered.iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.to_csv(output_dir / "selected_configs.csv", index=False)
    return runs, summary, selected


def run_confirmatory(
    *,
    base: torch.nn.Module,
    split,
    corruptions: dict[str, dict[str, object]],
    architecture: str,
    specs: tuple[MethodSpec, ...],
    selected: pd.DataFrame,
    confirmatory_seeds: tuple[int, ...],
    clean_metrics: dict[str, float],
    max_epochs: int,
    patience: int,
    output_dir: Path,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    spec_by_id = {spec.method_id: spec for spec in specs}
    for scenario_index, scenario in enumerate(SCENARIOS):
        x_val = corruptions[scenario]["val"]
        x_test = corruptions[scenario]["test"]
        frozen_validation = evaluate(base, x_val, split.y_val)
        frozen_target = evaluate(base, x_test, split.y_test)
        scenario_selection = selected[selected["scenario"] == scenario]
        for row in scenario_selection.itertuples(index=False):
            spec = spec_by_id[row.method_id]
            allocation = parse_allocation(row.allocation)
            learning_rate = float(row.learning_rate)
            for seed in confirmatory_seeds:
                if spec.implementation == "frozen":
                    model = base
                    training_info: dict[str, float | int] = {
                        "validation_accuracy": frozen_validation["accuracy"],
                        "validation_nll": frozen_validation["nll"],
                        "best_epoch": 0,
                        "epochs_ran": 0,
                        "trainable_parameters": 0,
                        "train_seconds": 0.0,
                    }
                    target_metrics = frozen_target
                    retained_metrics = clean_metrics
                else:
                    x_train, y_train = target_data(
                        split,
                        scenario,
                        scenario_index,
                        seed,
                    )
                    model, training_info = train_candidate(
                        base,
                        method=spec.implementation,
                        rank=allocation,
                        learning_rate=learning_rate,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=split.y_val,
                        seed=model_seed(architecture, scenario_index, seed),
                        max_epochs=max_epochs,
                        patience=patience,
                        lora_plus_ratio=LORA_PLUS_RATIO,
                    )
                    target_metrics = evaluate(model, x_test, split.y_test)
                    retained_metrics = evaluate(model, split.x_test, split.y_test)
                records.append(
                    {
                        "architecture": architecture,
                        "scenario": scenario,
                        "seed": seed,
                        "method_id": spec.method_id,
                        "method": spec.label,
                        "implementation": spec.implementation,
                        "allocation": allocation_text(allocation),
                        "selected_learning_rate": learning_rate,
                        "lora_plus_ratio": LORA_PLUS_RATIO if spec.method_id == "lora_plus" else 1.0,
                        **training_info,
                        "target_accuracy": target_metrics["accuracy"],
                        "target_macro_f1": target_metrics["macro_f1"],
                        "target_nll": target_metrics["nll"],
                        "clean_accuracy": retained_metrics["accuracy"],
                        "clean_macro_f1": retained_metrics["macro_f1"],
                        "clean_nll": retained_metrics["nll"],
                    }
                )
                save_records(records, output_dir / "confirmatory_runs.csv")
            print(
                f"confirmatory architecture={architecture} scenario={scenario} method={spec.method_id}",
                flush=True,
            )
    return pd.DataFrame(records)


def summarize_confirmatory(runs: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        runs.groupby(
            [
                "architecture",
                "scenario",
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
    summary.to_csv(output_dir / "confirmatory_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    output_dir = args.output_dir or ROOT / "results" / f"confirmatory_{args.architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory is not empty: {output_dir}. Use a fresh directory to preserve the frozen protocol."
        )

    split = load_split()
    pretrain_epochs = 40 if args.quick else 160
    max_epochs = 25 if args.quick else 120
    patience = 5 if args.quick else 18
    pilot_seeds = PILOT_SEEDS[:2] if args.quick else PILOT_SEEDS
    default_confirmatory = MLP_CONFIRMATORY_SEEDS if args.architecture == "mlp" else CNN_CONFIRMATORY_SEEDS
    confirmatory_seeds = default_confirmatory[:2] if args.quick else default_confirmatory
    specs = method_specs(args.architecture, quick=args.quick)

    started = time.time()
    base = pretrain_base(
        split,
        seed=2026,
        epochs=pretrain_epochs,
        architecture=args.architecture,
    )
    clean_metrics = evaluate(base, split.x_test, split.y_test)
    corruptions = fixed_corruptions(split)
    _, _, selected = run_pilot(
        base=base,
        split=split,
        corruptions=corruptions,
        architecture=args.architecture,
        specs=specs,
        pilot_seeds=pilot_seeds,
        max_epochs=max_epochs,
        patience=patience,
        output_dir=output_dir,
    )

    test_evaluated = False
    if not args.pilot_only:
        runs = run_confirmatory(
            base=base,
            split=split,
            corruptions=corruptions,
            architecture=args.architecture,
            specs=specs,
            selected=selected,
            confirmatory_seeds=confirmatory_seeds,
            clean_metrics=clean_metrics,
            max_epochs=max_epochs,
            patience=patience,
            output_dir=output_dir,
        )
        summary = summarize_confirmatory(runs, output_dir)
        print(summary.to_string(index=False), flush=True)
        test_evaluated = True

    metadata = {
        "protocol": "docs/EXTENSION_PROTOCOL.md",
        "architecture": args.architecture,
        "dataset": "sklearn.datasets.load_digits",
        "split_seed": 2026,
        "adaptation_examples": 400,
        "pilot_seeds": pilot_seeds,
        "confirmatory_seeds": confirmatory_seeds,
        "scenarios": SCENARIOS,
        "standard_learning_rates": STANDARD_LRS,
        "full_learning_rates": FULL_LRS,
        "lora_plus_a_learning_rates": LORA_PLUS_LRS,
        "lora_plus_b_over_a_ratio": LORA_PLUS_RATIO,
        "pretrain_epochs": pretrain_epochs,
        "max_adaptation_epochs": max_epochs,
        "patience": patience,
        "base_clean_test_metrics": clean_metrics,
        "test_evaluated_only_after_all_selections": test_evaluated,
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "quick": args.quick,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
