from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
DIGITS_DIR = ROOT / "results" / "digits"
SYNTHETIC_DIR = ROOT / "results" / "synthetic"


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, float("nan"), float("nan")
    standard_error = float(stats.sem(array))
    margin = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1) * standard_error)
    return mean, mean - margin, mean + margin


def validate_digits(runs: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    expected_scenarios = {"contrast", "rotation", "mixed"}
    expected_seeds = {11, 22, 33, 44, 55}
    if set(runs["scenario"]) != expected_scenarios:
        issues.append("unexpected scenario coverage")
    if set(runs["seed"]) != expected_seeds:
        issues.append("unexpected seed coverage")
    if not runs["target_accuracy"].between(0.0, 1.0).all():
        issues.append("target accuracy outside [0, 1]")
    if not runs["target_macro_f1"].between(0.0, 1.0).all():
        issues.append("macro-F1 outside [0, 1]")
    duplicate_count = runs.duplicated(["scenario", "seed", "method", "rank"]).sum()
    if duplicate_count:
        issues.append(f"{duplicate_count} duplicate method/rank runs")

    adapter = runs[runs["method"].isin(["LoRA", "DoRA"])]
    coverage = adapter.groupby(["scenario", "seed", "method"])["rank"].nunique()
    if not (coverage == 4).all():
        issues.append("incomplete rank coverage for LoRA/DoRA")
    if runs[["target_accuracy", "validation_accuracy", "trainable_parameters"]].isna().any().any():
        issues.append("missing key digit-benchmark values")
    return issues


def main() -> None:
    digit_runs = pd.read_csv(DIGITS_DIR / "digits_runs.csv")
    synthetic_runs = pd.read_csv(SYNTHETIC_DIR / "synthetic_runs.csv")
    issues = validate_digits(digit_runs)
    if issues:
        raise RuntimeError("validation failed: " + "; ".join(issues))

    paired = (
        digit_runs[digit_runs["method"].isin(["LoRA", "DoRA"])]
        .pivot(index=["scenario", "seed", "rank"], columns="method", values="target_accuracy")
        .reset_index()
    )
    paired["dora_minus_lora_pp"] = 100.0 * (paired["DoRA"] - paired["LoRA"])
    paired.to_csv(DIGITS_DIR / "paired_deltas.csv", index=False)

    paired_rows: list[dict[str, float | int | str]] = []
    for (scenario, rank), group in paired.groupby(["scenario", "rank"]):
        mean, lower, upper = mean_ci(group["dora_minus_lora_pp"])
        paired_rows.append(
            {
                "scenario": scenario,
                "rank": int(rank),
                "mean_delta_pp": mean,
                "ci95_low_pp": lower,
                "ci95_high_pp": upper,
                "std_delta_pp": float(group["dora_minus_lora_pp"].std()),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(DIGITS_DIR / "paired_delta_summary.csv", index=False)

    selected_rows: list[pd.Series] = []
    for _, group in digit_runs.groupby(["scenario", "seed", "method"]):
        ordered = group.sort_values(["validation_accuracy", "validation_nll"], ascending=[False, True])
        selected_rows.append(ordered.iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.to_csv(DIGITS_DIR / "validation_selected_runs.csv", index=False)

    selected_summary_rows: list[dict[str, float | int | str]] = []
    for (scenario, method), group in selected.groupby(["scenario", "method"]):
        mean, lower, upper = mean_ci(group["target_accuracy"] * 100.0)
        selected_summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "accuracy_mean_pct": mean,
                "accuracy_ci95_low_pct": lower,
                "accuracy_ci95_high_pct": upper,
                "accuracy_std_pct": float(group["target_accuracy"].std() * 100.0),
                "parameters_mean": float(group["trainable_parameters"].mean()),
                "selected_rank_median": float(group["rank"].median()),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    selected_summary = pd.DataFrame(selected_summary_rows)
    selected_summary.to_csv(DIGITS_DIR / "validation_selected_summary.csv", index=False)

    fixed_rank = digit_runs[
        ((digit_runs["method"].isin(["LoRA", "DoRA"])) & (digit_runs["rank"] == 4))
        | digit_runs["method"].isin(["Frozen", "Magnitude-only", "Full fine-tuning"])
    ]
    fixed_rank_summary_rows: list[dict[str, float | int | str]] = []
    for (scenario, method), group in fixed_rank.groupby(["scenario", "method"]):
        mean, lower, upper = mean_ci(group["target_accuracy"] * 100.0)
        fixed_rank_summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "accuracy_mean_pct": mean,
                "accuracy_ci95_low_pct": lower,
                "accuracy_ci95_high_pct": upper,
                "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    pd.DataFrame(fixed_rank_summary_rows).to_csv(DIGITS_DIR / "rank4_method_summary.csv", index=False)

    synthetic_paired = synthetic_runs.pivot_table(
        index=["seed", "rank", "magnitude_strength"],
        columns="method",
        values="relative_weight_error",
    ).reset_index()
    synthetic_paired["capacity_gap_log10"] = np.log10(
        synthetic_paired["LoRA SVD oracle"].clip(lower=1e-12)
        / synthetic_paired["DoRA construction"].clip(lower=1e-12)
    )
    synthetic_paired.to_csv(SYNTHETIC_DIR / "capacity_gap_runs.csv", index=False)
    capacity_summary = (
        synthetic_paired.groupby(["rank", "magnitude_strength"], as_index=False)
        .agg(
            capacity_gap_log10_mean=("capacity_gap_log10", "mean"),
            lora_relative_error_mean=("LoRA SVD oracle", "mean"),
            dora_relative_error_mean=("DoRA construction", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    capacity_summary.to_csv(SYNTHETIC_DIR / "capacity_gap_summary.csv", index=False)

    rank4_delta = paired_summary[paired_summary["rank"] == 4].set_index("scenario")
    mixed_selected = selected_summary[selected_summary["scenario"] == "mixed"].set_index("method")
    key_metrics = {
        "base_clean_accuracy_pct": 97.5,
        "rank4_dora_minus_lora_pp": rank4_delta["mean_delta_pp"].to_dict(),
        "rank4_delta_ci95_pp": {
            scenario: [float(row["ci95_low_pp"]), float(row["ci95_high_pp"])]
            for scenario, row in rank4_delta.iterrows()
        },
        "mixed_validation_selected_accuracy_pct": mixed_selected["accuracy_mean_pct"].to_dict(),
        "synthetic_rank4_lora_error_at_gamma_08": float(
            capacity_summary.query("rank == 4 and magnitude_strength == 0.8")["lora_relative_error_mean"].iloc[0]
        ),
        "synthetic_rank4_dora_error_at_gamma_08": float(
            capacity_summary.query("rank == 4 and magnitude_strength == 0.8")["dora_relative_error_mean"].iloc[0]
        ),
        "validation_status": "ready_with_caveats",
        "caveats": [
            "Small real-data proxy (sklearn Digits), not an LLM reproduction.",
            "Five paired adaptation seeds with one fixed pretrained backbone.",
            "Synthetic DoRA result is a feasible ground-truth construction; LoRA receives its exact SVD optimum.",
        ],
    }
    (ROOT / "results" / "key_metrics.json").write_text(json.dumps(key_metrics, indent=2), encoding="utf-8")
    print(json.dumps(key_metrics, indent=2))


if __name__ == "__main__":
    main()
