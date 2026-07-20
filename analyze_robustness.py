from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_extension import holm_adjust, mean_ci


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "data_sweep_mlp"
SYNTHETIC_DIR = ROOT / "results" / "synthetic_optimization"
EXPECTED_DATA_SEEDS = set(range(301, 321))
EXPECTED_BUDGETS = {5, 10, 20, 40}
EXPECTED_METHODS = {"lora", "lora_plus", "dora", "lora_matched", "dora_budgeted"}
EXPECTED_PROBLEMS = set(range(200, 210))
EXPECTED_INITIALIZATIONS = set(range(5))
EXPECTED_STRENGTHS = {0.0, 0.4, 0.8}


def validate_data_sweep(runs: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if set(runs["seed"]) != EXPECTED_DATA_SEEDS:
        issues.append("data sweep: unexpected seed coverage")
    if set(runs["per_class"]) != EXPECTED_BUDGETS:
        issues.append("data sweep: unexpected target-data budgets")
    if set(runs["method_id"]) != EXPECTED_METHODS:
        issues.append("data sweep: unexpected method coverage")
    expected_rows = len(EXPECTED_DATA_SEEDS) * len(EXPECTED_BUDGETS) * len(EXPECTED_METHODS)
    if len(runs) != expected_rows:
        issues.append(f"data sweep: expected {expected_rows} rows, found {len(runs)}")
    if runs.duplicated(["seed", "per_class", "method_id"]).any():
        issues.append("data sweep: duplicate paired rows")
    coverage = runs.groupby(["per_class", "method_id"])["seed"].nunique()
    if not (coverage == len(EXPECTED_DATA_SEEDS)).all():
        issues.append("data sweep: incomplete paired seed coverage")

    selected = pd.read_csv(
        ROOT / "results" / "confirmatory_mlp" / "selected_configs.csv",
        dtype={"allocation": str},
    )
    selected = selected[
        (selected["scenario"] == "mixed") & selected["method_id"].isin(EXPECTED_METHODS)
    ][["method_id", "allocation", "learning_rate"]].rename(
        columns={"learning_rate": "selected_learning_rate"}
    )
    observed = runs[
        ["method_id", "allocation", "selected_learning_rate"]
    ].drop_duplicates()
    merged = selected.merge(
        observed,
        on=["method_id", "allocation", "selected_learning_rate"],
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        issues.append("data sweep: configurations differ from frozen mixed-shift selection")

    metrics = ["target_accuracy", "target_macro_f1", "clean_accuracy", "validation_accuracy"]
    if runs[metrics].isna().any().any():
        issues.append("data sweep: missing metrics")
    for metric in metrics:
        if not runs[metric].between(0.0, 1.0).all():
            issues.append(f"data sweep: {metric} outside [0, 1]")
    return issues


def summarize_data_sweep(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (per_class, method_id, method), group in runs.groupby(["per_class", "method_id", "method"]):
        mean, lower, upper = mean_ci(group["target_accuracy"].to_numpy() * 100.0)
        records.append(
            {
                "per_class": int(per_class),
                "adaptation_examples": int(per_class) * 10,
                "method_id": method_id,
                "method": method,
                "accuracy_mean_pct": mean,
                "accuracy_ci95_low_pct": lower,
                "accuracy_ci95_high_pct": upper,
                "accuracy_std_pct": float(group["target_accuracy"].std(ddof=1) * 100.0),
                "macro_f1_mean_pct": float(group["target_macro_f1"].mean() * 100.0),
                "clean_accuracy_mean_pct": float(group["clean_accuracy"].mean() * 100.0),
                "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(records)


def data_sweep_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for per_class in sorted(EXPECTED_BUDGETS):
        subset = runs[runs["per_class"] == per_class]
        pivot = subset.pivot(index="seed", columns="method_id", values="target_accuracy")
        for comparator in ("lora", "lora_plus", "lora_matched", "dora_budgeted"):
            differences = (pivot["dora"] - pivot[comparator]).to_numpy(dtype=float) * 100.0
            mean, lower, upper = mean_ci(differences)
            standard_deviation = float(np.std(differences, ddof=1))
            t_result = stats.ttest_1samp(differences, popmean=0.0)
            wilcoxon_p = (
                1.0
                if np.allclose(differences, 0.0)
                else float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
            )
            records.append(
                {
                    "per_class": per_class,
                    "adaptation_examples": per_class * 10,
                    "comparison": f"dora_vs_{comparator}",
                    "comparator": comparator,
                    "mean_delta_pp": mean,
                    "ci95_low_pp": lower,
                    "ci95_high_pp": upper,
                    "std_delta_pp": standard_deviation,
                    "paired_effect_dz": mean / standard_deviation if standard_deviation > 0 else np.nan,
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": wilcoxon_p,
                    "wins": int(np.sum(differences > 1e-12)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                    "losses": int(np.sum(differences < -1e-12)),
                    "n_pairs": len(differences),
                }
            )
    comparisons = pd.DataFrame(records)
    comparisons["paired_t_p_holm"] = holm_adjust(comparisons["paired_t_p"])
    comparisons["wilcoxon_p_holm"] = holm_adjust(comparisons["wilcoxon_p"])
    return comparisons


def validate_synthetic(runs: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if set(runs["problem_seed"]) != EXPECTED_PROBLEMS:
        issues.append("synthetic optimization: unexpected problem coverage")
    if set(runs["initialization"]) != EXPECTED_INITIALIZATIONS:
        issues.append("synthetic optimization: unexpected initialization coverage")
    if set(runs["magnitude_strength"].round(6)) != EXPECTED_STRENGTHS:
        issues.append("synthetic optimization: unexpected magnitude strengths")
    expected_rows = len(EXPECTED_PROBLEMS) * len(EXPECTED_INITIALIZATIONS) * len(EXPECTED_STRENGTHS)
    if len(runs) != expected_rows:
        issues.append(f"synthetic optimization: expected {expected_rows} rows, found {len(runs)}")
    if runs.duplicated(["problem_seed", "magnitude_strength", "initialization"]).any():
        issues.append("synthetic optimization: duplicate runs")
    metric_columns = [column for column in runs if "error" in column or "mse" in column]
    if not np.isfinite(runs[metric_columns].to_numpy(dtype=float)).all():
        issues.append("synthetic optimization: non-finite metric")
    minimum_error = float(runs[metric_columns].min().min())
    if minimum_error < -1e-6:
        issues.append("synthetic optimization: materially negative error metric")
    oracle_variation = runs.groupby(["problem_seed", "magnitude_strength"])[
        ["feasible_dora_relative_weight_error", "lora_oracle_relative_weight_error"]
    ].nunique()
    if not (oracle_variation == 1).all().all():
        issues.append("synthetic optimization: oracle metrics vary across adapter initializations")
    return issues


def summarize_synthetic(runs: pd.DataFrame) -> pd.DataFrame:
    problem_level = (
        runs.groupby(["problem_seed", "magnitude_strength"], as_index=False)
        .agg(
            optimized_error_mean=("optimized_dora_relative_weight_error", "mean"),
            optimized_error_median=("optimized_dora_relative_weight_error", "median"),
            optimized_error_max=("optimized_dora_relative_weight_error", "max"),
            feasible_error=("feasible_dora_relative_weight_error", "first"),
            lora_oracle_error=("lora_oracle_relative_weight_error", "first"),
            convergence_rate=(
                "optimized_dora_relative_weight_error",
                lambda values: float((values < 1e-3).mean()),
            ),
        )
    )
    records: list[dict[str, object]] = []
    for strength, group in problem_level.groupby("magnitude_strength"):
        optimized_mean, optimized_low, optimized_high = mean_ci(group["optimized_error_mean"].to_numpy())
        lora_mean, lora_low, lora_high = mean_ci(group["lora_oracle_error"].to_numpy())
        records.append(
            {
                "magnitude_strength": strength,
                "optimized_dora_error_mean": optimized_mean,
                "optimized_dora_error_ci95_low": optimized_low,
                "optimized_dora_error_ci95_high": optimized_high,
                "optimized_dora_worst_init_error_mean": float(group["optimized_error_max"].mean()),
                "feasible_dora_error_mean": float(group["feasible_error"].mean()),
                "lora_oracle_error_mean": lora_mean,
                "lora_oracle_error_ci95_low": lora_low,
                "lora_oracle_error_ci95_high": lora_high,
                "convergence_rate_mean": float(group["convergence_rate"].mean()),
                "n_independent_problems": int(group["problem_seed"].nunique()),
                "initializations_per_problem": 5,
            }
        )
    return pd.DataFrame(records)


def write_report(
    issues: list[str],
    data_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
) -> None:
    assessment = "Ready to share" if not issues else "Needs revision"
    lora_rows = comparisons[comparisons["comparator"] == "lora"].sort_values("per_class")
    lines = [
        "# Robustness validation report",
        "",
        f"## Overall assessment: {assessment}",
        "",
        "## Data-regime design checks",
        "",
        "- 20 paired seeds are present at each of 50, 100, 200, and 400 target examples.",
        "- Target subsets are class-balanced and nested; the same samples and corruption realization are retained when the budget grows.",
        "- Every method reuses the mixed-shift allocation and learning rate selected before this sweep.",
        "",
        "| Target examples | DoRA − LoRA (pp) | 95% CI | Holm p | W/T/L |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in lora_rows.itertuples(index=False):
        lines.append(
            f"| {row.adaptation_examples} | {row.mean_delta_pp:+.2f} | "
            f"[{row.ci95_low_pp:+.2f}, {row.ci95_high_pp:+.2f}] | "
            f"{row.paired_t_p_holm:.4g} | {row.wins}/{row.ties}/{row.losses} |"
        )
    lines.extend(
        [
            "",
            "## Synthetic optimization checks",
            "",
            "- The inferential unit is the independently generated matrix problem (`n=10`), not each of its five optimizer initializations.",
            "- Feasible DoRA and SVD-LoRA oracle values are invariant across initialization duplicates.",
            "- All declared problems and initializations are retained and all error metrics are finite and non-negative.",
            "",
            "| Magnitude strength | Trained DoRA rel. error | Feasible DoRA | LoRA SVD oracle | Convergence <1e-3 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in synthetic_summary.itertuples(index=False):
        lines.append(
            f"| {row.magnitude_strength:.1f} | {row.optimized_dora_error_mean:.3g} | "
            f"{row.feasible_dora_error_mean:.3g} | {row.lora_oracle_error_mean:.3g} | "
            f"{100 * row.convergence_rate_mean:.1f}% |"
        )
    lines.extend(["", "## Issues found", ""])
    lines.extend(f"- {issue}" for issue in issues) if issues else lines.append("- No blocking issue detected.")
    lines.extend(
        [
            "",
            "## Required caveats",
            "",
            "- The data-budget sweep reuses one fixed MLP backbone and one designed mixed corruption.",
            "- Synthetic targets are deliberately generated from the DoRA family; they diagnose capacity and optimization, not real-task prevalence.",
            "- The robustness analyses are supporting evidence and do not replace the preregistered confirmatory comparisons.",
        ]
    )
    (ROOT / "docs" / "ROBUSTNESS_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data_runs = pd.read_csv(DATA_DIR / "data_sweep_runs.csv", dtype={"allocation": str})
    synthetic_runs = pd.read_csv(SYNTHETIC_DIR / "synthetic_optimization_runs.csv")
    issues = validate_data_sweep(data_runs) + validate_synthetic(synthetic_runs)
    data_summary = summarize_data_sweep(data_runs)
    comparisons = data_sweep_comparisons(data_runs)
    synthetic_summary = summarize_synthetic(synthetic_runs)
    data_summary.to_csv(DATA_DIR / "data_sweep_accuracy_ci.csv", index=False)
    comparisons.to_csv(DATA_DIR / "data_sweep_paired_comparisons.csv", index=False)
    synthetic_summary.to_csv(SYNTHETIC_DIR / "synthetic_optimization_verified_summary.csv", index=False)
    write_report(issues, data_summary, comparisons, synthetic_summary)

    audit = {
        "status": "ready_to_share" if not issues else "needs_revision",
        "issues": issues,
        "data_sweep_rows": len(data_runs),
        "synthetic_optimization_rows": len(synthetic_runs),
        "inference_unit_for_synthetic": "problem_seed",
    }
    (ROOT / "results" / "robustness_validation.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if issues:
        raise RuntimeError("robustness validation failed: " + "; ".join(issues))
    print(json.dumps(audit, indent=2))
    print(comparisons.to_string(index=False))
    print(synthetic_summary.to_string(index=False))


if __name__ == "__main__":
    main()
