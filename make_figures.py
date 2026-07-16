from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

INK = "#15242D"
MUTED = "#66747C"
GRID = "#DDE5E7"
TEAL = "#12A594"
BLUE = "#3565D4"
ORANGE = "#F39B42"
GREY = "#AAB4B9"
PALE = "#EEF4F4"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def synthetic_heatmap() -> None:
    data = pd.read_csv(ROOT / "results" / "synthetic" / "capacity_gap_summary.csv")
    matrix = data.pivot(index="rank", columns="magnitude_strength", values="capacity_gap_log10_mean").sort_index()
    values = np.maximum(matrix.to_numpy(), 0.0)
    cmap = LinearSegmentedColormap.from_list("airi_teal", ["#F4F7F7", "#9ADBD3", "#087B70"])

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=6.0, cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)), [f"{value:.1f}" for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [str(value) for value in matrix.index])
    ax.set_xlabel("Сила row-wise magnitude shift, γ")
    ax.set_ylabel("Adapter rank")
    ax.set_title("Разрыв выразительности: LoRA SVD / DoRA-конструкция", loc="left", fontweight="bold")
    ax.text(
        0.0,
        1.03,
        "log₁₀ отношения относительных ошибок; 10 задач на ячейку",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "≈0" if value < 0.05 else f"{value:.1f}"
            color = "white" if value > 3.6 else INK
            ax.text(column, row, label, ha="center", va="center", color=color, fontweight="bold", fontsize=11)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("порядки величины", color=MUTED)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.01,
        0.01,
        "Контроль: при γ=0 и rank≥4 оба метода представляют цель точно. Синтетическая capacity-задача, не accuracy-бенчмарк.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGURES / "synthetic_capacity_gap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "synthetic_capacity_gap.svg", bbox_inches="tight")
    plt.close(fig)


def rank4_benchmark() -> None:
    data = pd.read_csv(ROOT / "results" / "digits" / "rank4_method_summary.csv")
    scenarios = ["contrast", "rotation", "mixed"]
    methods = ["Frozen", "Magnitude-only", "LoRA", "DoRA", "Full fine-tuning"]
    labels = {"contrast": "Контраст", "rotation": "Поворот", "mixed": "Смешанный"}
    colors = {
        "Frozen": GREY,
        "Magnitude-only": ORANGE,
        "LoRA": BLUE,
        "DoRA": TEAL,
        "Full fine-tuning": INK,
    }

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    centers = np.arange(len(scenarios))
    width = 0.15
    for index, method in enumerate(methods):
        subset = data[data["method"] == method].set_index("scenario").reindex(scenarios)
        means = subset["accuracy_mean_pct"].to_numpy()
        lower = means - subset["accuracy_ci95_low_pct"].to_numpy()
        upper = subset["accuracy_ci95_high_pct"].to_numpy() - means
        positions = centers + (index - 2) * width
        bars = ax.bar(
            positions,
            means,
            width=width,
            color=colors[method],
            edgecolor="white",
            linewidth=0.8,
            label=method,
            yerr=np.vstack([lower, upper]),
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": MUTED},
        )
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 1.2, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(centers, [labels[item] for item in scenarios])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Few-shot адаптация на реальных изображениях", loc="left", fontweight="bold")
    ax.text(
        0.0,
        1.03,
        "sklearn Digits · 400 target-примеров · 5 paired seed · LoRA/DoRA rank 4",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGURES / "digits_rank4_benchmark.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "digits_rank4_benchmark.svg", bbox_inches="tight")
    plt.close(fig)


def paired_delta() -> None:
    data = pd.read_csv(ROOT / "results" / "digits" / "paired_delta_summary.csv")
    scenarios = ["contrast", "rotation", "mixed"]
    labels = {"contrast": "Контраст", "rotation": "Поворот", "mixed": "Смешанный"}

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        subset = data[data["scenario"] == scenario].sort_values("rank")
        mean = subset["mean_delta_pp"].to_numpy()
        lower = mean - subset["ci95_low_pp"].to_numpy()
        upper = subset["ci95_high_pp"].to_numpy() - mean
        ax.axhline(0.0, color=INK, linewidth=1)
        ax.errorbar(
            subset["rank"],
            mean,
            yerr=np.vstack([lower, upper]),
            color=TEAL,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=3,
        )
        ax.fill_between(subset["rank"], 0, mean, color=TEAL, alpha=0.08)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xlabel("Rank")
        ax.set_title(labels[scenario], fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_ylabel("DoRA − LoRA, п.п. accuracy")
    axes[0].set_ylim(-4.0, 6.0)
    fig.suptitle("Парная разница при одинаковом rank", x=0.07, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.07, 0.88, "Среднее и 95% t-интервал по 5 seed", color=MUTED, fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.86])
    fig.savefig(FIGURES / "digits_paired_delta.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "digits_paired_delta.svg", bbox_inches="tight")
    plt.close(fig)


def mixed_efficiency() -> None:
    runs = pd.read_csv(ROOT / "results" / "digits" / "digits_runs.csv")
    data = (
        runs[runs["scenario"] == "mixed"]
        .groupby(["method", "rank", "trainable_parameters"], as_index=False)
        .agg(accuracy=("target_accuracy", "mean"))
    )
    colors = {"LoRA": BLUE, "DoRA": TEAL, "Magnitude-only": ORANGE, "Full fine-tuning": INK, "Frozen": GREY}
    markers = {"LoRA": "s", "DoRA": "o", "Magnitude-only": "D", "Full fine-tuning": "P", "Frozen": "X"}

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for method, group in data.groupby("method"):
        x = group["trainable_parameters"].replace(0, 50)
        y = group["accuracy"] * 100.0
        ax.scatter(x, y, s=70, color=colors[method], marker=markers[method], label=method, edgecolor="white", linewidth=0.8)
        if method in {"LoRA", "DoRA"}:
            group_sorted = group.sort_values("rank")
            ax.plot(
                group_sorted["trainable_parameters"],
                group_sorted["accuracy"] * 100.0,
                color=colors[method],
                linewidth=1.5,
                alpha=0.8,
            )
            for _, row in group.iterrows():
                ax.annotate(f"r{int(row['rank'])}", (max(row["trainable_parameters"], 50), row["accuracy"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(35, 25000)
    ax.set_ylim(45, 80)
    ax.set_xlabel("Обучаемые параметры (log scale)")
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Эффективность на mixed shift", loc="left", fontweight="bold")
    ax.text(0.0, 1.03, "Среднее по 5 seed; frozen показан при x=50", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.grid(color=GRID, linewidth=0.8, which="both")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "mixed_efficiency_frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "mixed_efficiency_frontier.svg", bbox_inches="tight")
    plt.close(fig)


def poster_evidence_composite() -> None:
    capacity = pd.read_csv(ROOT / "results" / "synthetic" / "capacity_gap_summary.csv")
    capacity = capacity[capacity["rank"] == 4].sort_values("magnitude_strength")
    paired = pd.read_csv(ROOT / "results" / "digits" / "paired_delta_summary.csv")
    paired = paired[paired["rank"] == 4].set_index("scenario").reindex(["contrast", "rotation", "mixed"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.75), gridspec_kw={"width_ratios": [1.04, 0.96]})
    left, right = axes

    left.plot(
        capacity["magnitude_strength"],
        capacity["lora_relative_error_mean"],
        color=BLUE,
        marker="s",
        markersize=6,
        linewidth=2.2,
        label="LoRA SVD optimum",
    )
    left.plot(
        capacity["magnitude_strength"],
        capacity["dora_relative_error_mean"],
        color=TEAL,
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="DoRA construction",
    )
    left.set_yscale("log")
    left.set_ylim(5e-8, 1.0)
    left.set_xticks(capacity["magnitude_strength"])
    left.set_xlabel("Magnitude shift, γ")
    left.set_ylabel("Относительная ошибка весов")
    left.set_title("A. Контролируемая capacity-задача", loc="left", fontweight="bold", pad=18)
    left.text(0.0, 0.955, "rank 4 · 10 задач на точку", transform=left.transAxes, color=MUTED, fontsize=9)
    left.grid(color=GRID, linewidth=0.8, which="both")
    left.legend(frameon=False, fontsize=8.5, loc="center left")

    mean = paired["mean_delta_pp"].to_numpy()
    lower = mean - paired["ci95_low_pp"].to_numpy()
    upper = paired["ci95_high_pp"].to_numpy() - mean
    positions = np.arange(3)
    right.axhline(0.0, color=INK, linewidth=1)
    right.errorbar(
        positions,
        mean,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=8,
        color=TEAL,
        ecolor=TEAL,
        elinewidth=2,
        capsize=4,
    )
    right.set_xticks(positions, ["Контраст", "Поворот", "Смешанный"])
    right.set_ylim(-2.3, 3.4)
    right.set_ylabel("DoRA − LoRA, п.п. accuracy")
    right.set_title("B. Реальные изображения", loc="left", fontweight="bold", pad=18)
    right.text(0.0, 0.955, "rank 4 · среднее и 95% CI · 5 seed", transform=right.transAxes, color=MUTED, fontsize=9)
    right.grid(axis="y", color=GRID, linewidth=0.8)
    for x, value in zip(positions, mean):
        right.annotate(f"{value:+.2f}", (x, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(FIGURES / "poster_evidence_composite.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "poster_evidence_composite.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    synthetic_heatmap()
    rank4_benchmark()
    paired_delta()
    mixed_efficiency()
    poster_evidence_composite()
    print(f"saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
