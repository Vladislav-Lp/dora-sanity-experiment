from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCENARIOS = {"contrast", "rotation", "mixed"}
EXPECTED = {
    "mlp": {
        "directory": RESULTS / "confirmatory_mlp",
        "seeds": set(range(101, 121)),
        "pilot_seeds": {11, 22, 33, 44, 55},
        "methods": {
            "frozen",
            "magnitude",
            "lora",
            "lora_plus",
            "dora",
            "full",
            "lora_matched",
            "dora_budgeted",
        },
    },
    "cnn": {
        "directory": RESULTS / "confirmatory_cnn",
        "seeds": set(range(201, 211)),
        "pilot_seeds": {11, 22, 33, 44, 55},
        "methods": {"frozen", "magnitude", "lora", "lora_plus", "dora", "full"},
    },
}


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    margin = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * stats.sem(values))
    return mean, mean - margin, mean + margin


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(order)
    for position, original_index in enumerate(order):
        running = max(running, (m - position) * values[original_index])
        adjusted[original_index] = min(1.0, running)
    return pd.Series(adjusted, index=p_values.index)


def validation_issues(
    architecture: str,
    pilot: pd.DataFrame,
    selected: pd.DataFrame,
    confirmatory: pd.DataFrame,
) -> list[str]:
    expected = EXPECTED[architecture]
    issues: list[str] = []
    if set(pilot["scenario"]) != SCENARIOS or set(confirmatory["scenario"]) != SCENARIOS:
        issues.append(f"{architecture}: incomplete scenario coverage")
    if set(pilot["pilot_seed"]) != expected["pilot_seeds"]:
        issues.append(f"{architecture}: unexpected pilot seeds")
    if set(confirmatory["seed"]) != expected["seeds"]:
        issues.append(f"{architecture}: unexpected confirmatory seeds")
    if set(confirmatory["method_id"]) != expected["methods"]:
        issues.append(f"{architecture}: incomplete method coverage")
    if any(column.startswith("target_") for column in pilot.columns):
        issues.append(f"{architecture}: pilot file contains target-test metrics")

    pilot_coverage = pilot.groupby(
        ["scenario", "method_id", "allocation", "learning_rate"], dropna=False
    )["pilot_seed"].nunique()
    if not (pilot_coverage == len(expected["pilot_seeds"])).all():
        issues.append(f"{architecture}: a pilot candidate is missing seeds")
    confirmatory_coverage = confirmatory.groupby(["scenario", "method_id"])["seed"].nunique()
    if not (confirmatory_coverage == len(expected["seeds"])).all():
        issues.append(f"{architecture}: a confirmatory method is missing seeds")

    duplicate_keys = ["scenario", "seed", "method_id"]
    duplicate_count = int(confirmatory.duplicated(duplicate_keys).sum())
    if duplicate_count:
        issues.append(f"{architecture}: {duplicate_count} duplicate confirmatory rows")
    if selected.duplicated(["scenario", "method_id"]).any():
        issues.append(f"{architecture}: selected configuration is not unique")

    selected_keys = selected[["scenario", "method_id", "allocation", "learning_rate"]].rename(
        columns={"learning_rate": "selected_learning_rate"}
    )
    observed_keys = confirmatory[
        ["scenario", "method_id", "allocation", "selected_learning_rate"]
    ].drop_duplicates()
    merged = selected_keys.merge(
        observed_keys,
        on=["scenario", "method_id", "allocation", "selected_learning_rate"],
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        issues.append(f"{architecture}: confirmatory configuration differs from validation selection")

    numeric_metrics = [
        "validation_accuracy",
        "target_accuracy",
        "target_macro_f1",
        "clean_accuracy",
        "clean_macro_f1",
    ]
    if confirmatory[numeric_metrics].isna().any().any():
        issues.append(f"{architecture}: missing metric values")
    for metric in numeric_metrics:
        if not confirmatory[metric].between(0.0, 1.0).all():
            issues.append(f"{architecture}: {metric} outside [0, 1]")
    if (confirmatory["trainable_parameters"] < 0).any():
        issues.append(f"{architecture}: negative parameter count")
    return issues


def accuracy_summary(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (architecture, scenario, method_id, method), group in runs.groupby(
        ["architecture", "scenario", "method_id", "method"]
    ):
        mean, lower, upper = mean_ci(group["target_accuracy"].to_numpy() * 100.0)
        records.append(
            {
                "architecture": architecture,
                "scenario": scenario,
                "method_id": method_id,
                "method": method,
                "accuracy_mean_pct": mean,
                "accuracy_ci95_low_pct": lower,
                "accuracy_ci95_high_pct": upper,
                "accuracy_std_pct": float(group["target_accuracy"].std(ddof=1) * 100.0),
                "macro_f1_mean_pct": float(group["target_macro_f1"].mean() * 100.0),
                "target_nll_mean": float(group["target_nll"].mean()),
                "clean_accuracy_mean_pct": float(group["clean_accuracy"].mean() * 100.0),
                "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
                "allocation": group["allocation"].iloc[0],
                "selected_learning_rate": float(group["selected_learning_rate"].iloc[0]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(records)


def comparison_record(
    runs: pd.DataFrame,
    *,
    architecture: str,
    scenario: str,
    numerator: str,
    comparator: str,
    family: str,
) -> dict[str, object]:
    subset = runs[
        (runs["architecture"] == architecture)
        & (runs["scenario"] == scenario)
        & (runs["method_id"].isin([numerator, comparator]))
    ]
    paired = subset.pivot(index="seed", columns="method_id", values="target_accuracy")
    if numerator not in paired or comparator not in paired or paired.isna().any().any():
        raise RuntimeError(f"incomplete pairing: {architecture}/{scenario}/{numerator}/{comparator}")
    differences = (paired[numerator] - paired[comparator]).to_numpy(dtype=float) * 100.0
    mean, lower, upper = mean_ci(differences)
    standard_deviation = float(np.std(differences, ddof=1))
    effect_size = mean / standard_deviation if standard_deviation > 0 else float("nan")
    t_result = stats.ttest_1samp(differences, popmean=0.0)
    if np.allclose(differences, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    tolerance = 1e-12
    return {
        "architecture": architecture,
        "scenario": scenario,
        "comparison": f"{numerator}_vs_{comparator}",
        "numerator": numerator,
        "comparator": comparator,
        "family": family,
        "mean_delta_pp": mean,
        "ci95_low_pp": lower,
        "ci95_high_pp": upper,
        "std_delta_pp": standard_deviation,
        "paired_effect_dz": effect_size,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "wins": int(np.sum(differences > tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "losses": int(np.sum(differences < -tolerance)),
        "n_pairs": len(differences),
    }


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scenario in ("contrast", "rotation", "mixed"):
        for comparator in ("lora", "lora_plus", "lora_matched"):
            records.append(
                comparison_record(
                    runs,
                    architecture="mlp",
                    scenario=scenario,
                    numerator="dora",
                    comparator=comparator,
                    family="mlp_primary",
                )
            )
        records.append(
            comparison_record(
                runs,
                architecture="mlp",
                scenario=scenario,
                numerator="dora_budgeted",
                comparator="lora",
                family="mlp_secondary",
            )
        )
        for comparator in ("lora", "lora_plus"):
            records.append(
                comparison_record(
                    runs,
                    architecture="cnn",
                    scenario=scenario,
                    numerator="dora",
                    comparator=comparator,
                    family="cnn_replication",
                )
            )
    comparisons = pd.DataFrame(records)
    comparisons["paired_t_p_holm"] = np.nan
    comparisons["wilcoxon_p_holm"] = np.nan
    for family, indices in comparisons.groupby("family").groups.items():
        if family == "mlp_secondary":
            continue
        comparisons.loc[indices, "paired_t_p_holm"] = holm_adjust(
            comparisons.loc[indices, "paired_t_p"]
        )
        comparisons.loc[indices, "wilcoxon_p_holm"] = holm_adjust(
            comparisons.loc[indices, "wilcoxon_p"]
        )
    return comparisons


def write_validation_report(
    *,
    issues: list[str],
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    assessment = "Ready to share" if not issues else "Needs revision"
    main = comparisons[
        (comparisons["architecture"] == "mlp")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["numerator"] == "dora")
        & (comparisons["comparator"].isin(["lora", "lora_plus", "lora_matched"]))
    ]
    lines = [
        "# Extension validation report",
        "",
        f"## Overall assessment: {assessment}",
        "",
        "## Methodology and data-integrity checks",
        "",
        f"- Confirmatory rows checked: {len(runs):,}.",
        "- Pilot tables contain validation metrics only; target-test columns are rejected by the validator.",
        "- Every confirmatory row is paired by scenario and seed, and its allocation/LR is reconciled with the frozen selection table.",
        "- Accuracy, macro-F1, NLL, parameter counts, duplicate keys, method coverage, and seed coverage are checked from raw rows.",
        "",
        "## Issues found",
        "",
    ]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No blocking calculation or coverage issue detected.")
    lines.extend(
        [
            "",
            "## Primary mixed-shift spot checks",
            "",
            "| Comparison | Mean Δ (pp) | 95% CI | Holm p (paired t) | W/T/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in main.itertuples(index=False):
        lines.append(
            f"| DoRA − {row.comparator} | {row.mean_delta_pp:+.2f} | "
            f"[{row.ci95_low_pp:+.2f}, {row.ci95_high_pp:+.2f}] | "
            f"{row.paired_t_p_holm:.4g} | {row.wins}/{row.ties}/{row.losses} |"
        )
    lines.extend(
        [
            "",
            "## Required caveats",
            "",
            "- Digits is a small vision proxy, not an LLM-scale reproduction.",
            "- The extension uses one fixed pretrained backbone per architecture; adaptation seeds quantify few-shot sampling and optimization variability, not pretraining variability.",
            "- The three corruption scenarios are designed domain shifts; conclusions should be framed as geometry-dependent rather than universal superiority.",
            "- The same target split appeared in the earlier exploratory phase. Hyperparameters and new seed ranges were frozen before the extension, but this is a protocol-frozen internal confirmation rather than a fully untouched external replication.",
            "",
            "## Reproducible evidence",
            "",
            "- `results/extension_accuracy_summary.csv`",
            "- `results/extension_paired_comparisons.csv`",
            "- `results/confirmatory_mlp/selected_configs.csv`",
            "- `results/confirmatory_cnn/selected_configs.csv`",
        ]
    )
    (ROOT / "docs" / "EXTENSION_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_runs: list[pd.DataFrame] = []
    issues: list[str] = []
    for architecture, expected in EXPECTED.items():
        directory = expected["directory"]
        pilot = pd.read_csv(directory / "pilot_runs.csv", dtype={"allocation": str})
        selected = pd.read_csv(directory / "selected_configs.csv", dtype={"allocation": str})
        confirmatory = pd.read_csv(directory / "confirmatory_runs.csv", dtype={"allocation": str})
        issues.extend(validation_issues(architecture, pilot, selected, confirmatory))
        all_runs.append(confirmatory)

    runs = pd.concat(all_runs, ignore_index=True)
    summary = accuracy_summary(runs)
    comparisons = paired_comparisons(runs)
    summary.to_csv(RESULTS / "extension_accuracy_summary.csv", index=False)
    comparisons.to_csv(RESULTS / "extension_paired_comparisons.csv", index=False)
    write_validation_report(issues=issues, runs=runs, summary=summary, comparisons=comparisons)

    audit = {
        "status": "ready_to_share" if not issues else "needs_revision",
        "issues": issues,
        "confirmatory_rows": len(runs),
        "architectures": sorted(runs["architecture"].unique().tolist()),
        "scenarios": sorted(runs["scenario"].unique().tolist()),
        "generated_files": [
            "results/extension_accuracy_summary.csv",
            "results/extension_paired_comparisons.csv",
            "docs/EXTENSION_VALIDATION.md",
        ],
    }
    (RESULTS / "extension_validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if issues:
        raise RuntimeError("extension validation failed: " + "; ".join(issues))
    print(json.dumps(audit, indent=2))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
