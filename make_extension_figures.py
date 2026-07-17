from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures" / "extension"
FIGURES.mkdir(parents=True, exist_ok=True)

INK = "#15242D"
MUTED = "#66747C"
GRID = "#DDE5E7"
TEAL = "#0B8F83"
TEAL_LIGHT = "#9ADBD3"
BLUE = "#3565D4"
BLUE_LIGHT = "#AFC2EF"
ORANGE = "#E38A33"
GREY = "#AAB4B9"
PALE = "#F3F7F7"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def confirmatory_deltas() -> None:
    comparisons = pd.read_csv(ROOT / "results" / "extension_paired_comparisons.csv")
    data = comparisons[comparisons["comparison"] == "dora_vs_lora"].copy()
    scenario_order = ["contrast", "rotation", "mixed"]
    labels = {"contrast": "Contrast", "rotation": "Rotation", "mixed": "Mixed"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55), sharex=True, sharey=True)
    for ax, architecture, n_pairs, marker in zip(axes, ["mlp", "cnn"], [20, 10], ["o", "s"]):
        subset = data[data["architecture"] == architecture].set_index("scenario").reindex(scenario_order)
        y = np.arange(len(scenario_order))[::-1]
        means = subset["mean_delta_pp"].to_numpy()
        lower = means - subset["ci95_low_pp"].to_numpy()
        upper = subset["ci95_high_pp"].to_numpy() - means
        ax.axvline(0.0, color=INK, linewidth=1.1)
        ax.errorbar(
            means,
            y,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            color=TEAL,
            ecolor=TEAL,
            markerfacecolor="white" if architecture == "cnn" else TEAL,
            markeredgewidth=1.8,
            markersize=7,
            elinewidth=2,
            capsize=4,
            zorder=3,
        )
        ax.set_yticks(y, [labels[item] for item in scenario_order])
        ax.set_title(f"{architecture.upper()} · n={n_pairs} paired seeds", loc="left", fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for scenario, x, yy in zip(scenario_order, means, y):
            ax.annotate(
                f"{x:+.2f}",
                (x, yy),
                xytext=(0, -15 if scenario == "contrast" else 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    axes[0].set_xlim(-1.5, 2.6)
    for ax in axes:
        ax.set_xlabel("DoRA − LoRA test accuracy (percentage points)")
    fig.suptitle("Confirmatory paired effects across two backbones", x=0.07, ha="left", fontweight="bold")
    fig.text(
        0.07,
        0.90,
        "Mean and 95% paired t interval; configurations selected on validation before test evaluation",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.86], w_pad=2.0)
    save(fig, "confirmatory_dora_minus_lora")


def mixed_comparators() -> None:
    comparisons = pd.read_csv(ROOT / "results" / "extension_paired_comparisons.csv")
    data = comparisons[
        (comparisons["architecture"] == "mlp")
        & (comparisons["scenario"] == "mixed")
        & (comparisons["numerator"] == "dora")
        & comparisons["comparator"].isin(["lora", "lora_plus", "lora_matched"])
    ].copy()
    order = ["lora", "lora_plus", "lora_matched"]
    labels = {
        "lora": "LoRA · 1,832 params",
        "lora_plus": "LoRA+ · 1,832 params",
        "lora_matched": "LoRA budget-matched · 2,024 params",
    }
    data = data.set_index("comparator").reindex(order)
    y = np.arange(len(order))[::-1]
    means = data["mean_delta_pp"].to_numpy()
    lower = means - data["ci95_low_pp"].to_numpy()
    upper = data["ci95_high_pp"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axvline(0.0, color=INK, linewidth=1.1)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=TEAL,
        ecolor=TEAL,
        markersize=8,
        elinewidth=2.2,
        capsize=4,
    )
    ax.set_yticks(y, [labels[item] for item in order])
    ax.set_ylim(-0.3, 2.55)
    ax.set_xlim(-1.0, 3.0)
    ax.set_xlabel("DoRA r=4 (2,034 params) minus comparator accuracy, pp")
    ax.set_title(
        "MLP mixed shift: stronger and budget-matched baselines",
        loc="left",
        fontweight="bold",
        pad=28,
    )
    ax.text(
        0.0,
        1.01,
        "20 paired confirmatory seeds · 95% t intervals · Holm family includes 9 primary tests",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for index, (x, yy, p_value) in enumerate(zip(means, y, data["paired_t_p_holm"])):
        ax.annotate(
            f"{x:+.2f} pp",
            (x, yy),
            xytext=(0, -16 if index == 0 else 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        ax.text(2.95, yy, f"Holm p={p_value:.3f}", ha="right", va="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    save(fig, "mixed_strong_baselines")


def data_regime_accuracy() -> None:
    summary = pd.read_csv(ROOT / "results" / "data_sweep_mlp" / "data_sweep_accuracy_ci.csv")
    methods = ["dora", "lora", "lora_plus"]
    styles = {
        "dora": {"color": TEAL, "marker": "o", "linestyle": "-", "label": "DoRA"},
        "lora": {"color": BLUE, "marker": "s", "linestyle": "-", "label": "LoRA"},
        "lora_plus": {"color": BLUE, "marker": "o", "linestyle": "--", "label": "LoRA+"},
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for method in methods:
        subset = summary[summary["method_id"] == method].sort_values("adaptation_examples")
        x = subset["adaptation_examples"].to_numpy()
        mean = subset["accuracy_mean_pct"].to_numpy()
        lower = mean - subset["accuracy_ci95_low_pct"].to_numpy()
        upper = subset["accuracy_ci95_high_pct"].to_numpy() - mean
        style = styles[method]
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([lower, upper]),
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white" if method == "lora_plus" else style["color"],
            markeredgewidth=1.5,
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6,
            capsize=3,
            label=style["label"],
        )
    ax.set_xticks([50, 100, 200, 400])
    ax.set_ylim(56, 78)
    ax.set_xlabel("Target adaptation examples (balanced across 10 classes)")
    ax.set_ylabel("Mixed-shift test accuracy, %")
    ax.set_title("Target-data regime sweep", loc="left", fontweight="bold", pad=28)
    ax.text(
        0.0,
        1.01,
        "20 paired seeds · nested subsets · mean and 95% CI · focused y-axis",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
    )
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    save(fig, "data_regime_accuracy")


def data_regime_deltas() -> None:
    comparisons = pd.read_csv(
        ROOT / "results" / "data_sweep_mlp" / "data_sweep_paired_comparisons.csv"
    )
    styles = {
        "lora": {"label": "vs LoRA", "color": TEAL, "marker": "o", "offset": -5},
        "lora_plus": {"label": "vs LoRA+", "color": BLUE, "marker": "s", "offset": 5},
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axhline(0.0, color=INK, linewidth=1.1)
    for comparator, style in styles.items():
        subset = comparisons[comparisons["comparator"] == comparator].sort_values("adaptation_examples")
        x = subset["adaptation_examples"].to_numpy() + style["offset"]
        mean = subset["mean_delta_pp"].to_numpy()
        lower = mean - subset["ci95_low_pp"].to_numpy()
        upper = subset["ci95_high_pp"].to_numpy() - mean
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([lower, upper]),
            fmt=style["marker"],
            color=style["color"],
            markerfacecolor="white" if comparator == "lora_plus" else style["color"],
            markeredgewidth=1.6,
            markersize=7,
            elinewidth=2,
            capsize=3,
            label=style["label"],
        )
    ax.set_xticks([50, 100, 200, 400])
    ax.set_ylim(-1.5, 5.0)
    ax.set_xlabel("Target adaptation examples")
    ax.set_ylabel("DoRA accuracy advantage, pp")
    ax.set_title("Paired DoRA effects across data budgets", loc="left", fontweight="bold", pad=28)
    ax.text(
        0.0,
        1.01,
        "Mean and 95% paired t interval · n=20 seeds per budget",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "data_regime_paired_deltas")


def synthetic_optimization() -> None:
    runs = pd.read_csv(
        ROOT / "results" / "synthetic_optimization" / "synthetic_optimization_runs.csv"
    )
    problem = (
        runs.groupby(["problem_seed", "magnitude_strength"], as_index=False)
        .agg(
            trained=("optimized_dora_relative_weight_error", "mean"),
            feasible=("feasible_dora_relative_weight_error", "first"),
            lora=("lora_oracle_relative_weight_error", "first"),
        )
    )
    methods = [
        ("trained", "DoRA trained", TEAL, "o", -0.07, True),
        ("feasible", "DoRA feasible", TEAL_LIGHT, "o", 0.0, False),
        ("lora", "LoRA SVD oracle", BLUE, "s", 0.07, True),
    ]
    rng = np.random.default_rng(2026)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for column, label, color, marker, offset, filled in methods:
        for strength in [0.0, 0.4, 0.8]:
            values = problem[problem["magnitude_strength"] == strength][column].to_numpy()
            jitter = rng.uniform(-0.016, 0.016, size=len(values))
            ax.scatter(
                np.full(len(values), strength + offset) + jitter,
                values,
                s=30,
                color=color if filled else "white",
                edgecolor=color,
                marker=marker,
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot(
                [strength + offset - 0.027, strength + offset + 0.027],
                [median, median],
                color=INK,
                linewidth=1.5,
                zorder=4,
            )
    ax.set_yscale("log")
    ax.set_ylim(5e-8, 1.0)
    ax.set_xticks([0.0, 0.4, 0.8])
    ax.set_xlabel("Row-wise magnitude strength, γ")
    ax.set_ylabel("Relative weight error (log scale)")
    ax.set_title(
        "Synthetic capacity and optimization diagnostic",
        loc="left",
        fontweight="bold",
        pad=28,
    )
    ax.text(
        0.0,
        1.01,
        "rank 4 · 10 independent problems · trained points average 5 initializations · bar = median",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.9,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8, which="both")
    handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=color if filled else "white", markeredgecolor=color, label=label, markersize=7)
        for _, label, color, marker, _, filled in methods
    ]
    ax.legend(handles=handles, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(0.02, 0.60))
    fig.tight_layout()
    save(fig, "synthetic_optimization")


def mixed_accuracy_by_backbone() -> None:
    summary = pd.read_csv(ROOT / "results" / "extension_accuracy_summary.csv")
    data = summary[
        (summary["scenario"] == "mixed")
        & summary["method_id"].isin(["dora", "lora", "lora_plus", "full", "magnitude"])
    ].copy()
    order = ["dora", "lora", "lora_plus", "full", "magnitude"]
    labels = {
        "dora": "DoRA",
        "lora": "LoRA",
        "lora_plus": "LoRA+",
        "full": "Full FT",
        "magnitude": "Magnitude-only",
    }
    colors = {"dora": TEAL, "lora": BLUE, "lora_plus": BLUE_LIGHT, "full": INK, "magnitude": GREY}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
    for ax, architecture, n in zip(axes, ["mlp", "cnn"], [20, 10]):
        subset = data[data["architecture"] == architecture].set_index("method_id").reindex(order)
        y = np.arange(len(order))[::-1]
        means = subset["accuracy_mean_pct"].to_numpy()
        lower = means - subset["accuracy_ci95_low_pct"].to_numpy()
        upper = subset["accuracy_ci95_high_pct"].to_numpy() - means
        for index, method_id in enumerate(order):
            ax.errorbar(
                means[index],
                y[index],
                xerr=np.array([[lower[index]], [upper[index]]]),
                fmt="o",
                color=colors[method_id],
                markersize=7,
                elinewidth=2,
                capsize=3,
            )
        ax.set_yticks(y, [labels[item] for item in order])
        ax.set_title(f"{architecture.upper()} · n={n}", loc="left", fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for x, yy in zip(means, y):
            ax.annotate(f"{x:.1f}", (x, yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5)
    axes[0].set_xlim(62, 83)
    for ax in axes:
        ax.set_xlabel("Mixed-shift test accuracy, % (focused scale)")
    fig.suptitle("Mixed-shift accuracy by backbone", x=0.07, ha="left", fontweight="bold")
    fig.text(0.07, 0.90, "Mean and 95% CI across confirmatory seeds", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=[0, 0.01, 1, 0.86], w_pad=2.0)
    save(fig, "mixed_accuracy_by_backbone")


def main() -> None:
    setup()
    confirmatory_deltas()
    mixed_comparators()
    data_regime_accuracy()
    data_regime_deltas()
    synthetic_optimization()
    mixed_accuracy_by_backbone()
    print(f"saved extension figures to {FIGURES}")


if __name__ == "__main__":
    main()
