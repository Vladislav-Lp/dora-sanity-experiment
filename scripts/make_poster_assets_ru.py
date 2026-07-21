from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qrcode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster" / ".build" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#15242D"
MUTED = "#66747C"
GRID = "#DDE5E7"
TEAL = "#0B8F83"
BLUE = "#3565D4"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 11,
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

scenarios = ["Контраст", "Поворот", "Комбинированный"]
mlp_mean = np.array([-0.1944444444, 0.4166666667, 1.0555555556])
mlp_low = np.array([-0.5373666814, -0.2872277463, -0.0664340421])
mlp_high = np.array([0.1484777925, 1.1205610796, 2.1775451532])
cnn_mean = np.array([-0.3055555556, 0.3611111111, 0.9166666667])
cnn_low = np.array([-1.1138067731, -0.3826895533, 0.0997765801])
cnn_high = np.array([0.5026956620, 1.1049117755, 1.7335567532])

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharex=True, sharey=True)
y = np.arange(3)[::-1]
for ax, title, means, lows, highs, marker, filled in [
    (axes[0], "MLP · n=20 отложенных сидов", mlp_mean, mlp_low, mlp_high, "o", True),
    (axes[1], "CNN · n=10 отложенных сидов", cnn_mean, cnn_low, cnn_high, "s", False),
]:
    ax.axvline(0, color=INK, linewidth=1.2)
    lower = means - lows
    upper = highs - means
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt=marker,
        color=TEAL,
        ecolor=TEAL,
        markerfacecolor=TEAL if filled else "white",
        markeredgewidth=2,
        markersize=8,
        elinewidth=2.3,
        capsize=5,
        zorder=3,
    )
    ax.set_yticks(y, scenarios)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=15)
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for scenario, value, yy in zip(scenarios, means, y):
        ax.annotate(
            f"{value:+.2f}",
            (value, yy),
            xytext=(0, -18 if scenario == "Контраст" else 11),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
        )
    ax.set_xlabel("Разность точности DoRA − LoRA, п.п.")
axes[0].set_xlim(-1.45, 2.55)
fig.suptitle("Парные оценки на отложенных сидах", x=0.07, ha="left", fontsize=18, fontweight="bold")
fig.text(
    0.07,
    0.90,
    "Среднее и 95%-й парный t-интервал; один фиксированный чекпоинт на архитектуру",
    color=MUTED,
    fontsize=10.5,
)
fig.tight_layout(rect=[0, 0.01, 1, 0.84], w_pad=2.5)
fig.savefig(OUT / "paired_estimates_ru.png", dpi=300, bbox_inches="tight")
plt.close(fig)

x = np.array([50, 100, 200, 400])
series = {
    "DoRA": (
        np.array([61.4722222222, 65.8194444444, 70.6527777778, 74.8611111111]),
        np.array([60.5066116810, 64.5693016511, 69.4573398139, 73.8483790150]),
        np.array([62.4378327635, 67.0695872378, 71.8482157416, 75.8738432073]),
        TEAL,
        "o",
        "-",
    ),
    "LoRA": (
        np.array([61.0138888889, 65.2916666667, 69.7361111111, 74.1666666667]),
        np.array([60.1410053968, 64.4185793846, 68.6142206195, 72.9992945839]),
        np.array([61.8867723809, 66.1647539487, 70.8580016027, 75.3340387495]),
        BLUE,
        "s",
        "-",
    ),
    "LoRA+": (
        np.array([59.0138888889, 62.9861111111, 68.9027777778, 73.3194444444]),
        np.array([58.1357222267, 61.8552178696, 67.9395884438, 72.3325355468]),
        np.array([59.8920555510, 64.1170043526, 69.8659671117, 74.3063533421]),
        BLUE,
        "o",
        "--",
    ),
}
fig, ax = plt.subplots(figsize=(10.4, 5.2))
for label, (mean, low, high, color, marker, line_style) in series.items():
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - low, high - mean]),
        color=color,
        marker=marker,
        markerfacecolor="white" if label == "LoRA+" else color,
        markeredgewidth=1.8,
        linestyle=line_style,
        linewidth=2.4,
        markersize=8,
        capsize=4,
        label=label,
    )
ax.set_xticks(x)
ax.set_ylim(56, 78)
ax.set_xlabel("Целевые примеры для адаптации (баланс по 10 классам)")
ax.set_ylabel("Точность при комбинированном сдвиге, %")
ax.set_title("Точность при разных объёмах целевых данных", loc="left", fontweight="bold", pad=30, fontsize=18)
ax.text(
    0.0,
    1.01,
    "20 парных сидов · вложенные подвыборки · среднее и 95%-й ДИ",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=10.5,
)
ax.grid(color=GRID, linewidth=0.9)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "data_sweep_ru.png", dpi=300, bbox_inches="tight")
plt.close(fig)

url = "https://github.com/Vladislav-Lp/dora-sanity-experiment"
qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=16, border=2)
qr.add_data(url)
qr.make(fit=True)
qr.make_image(fill_color="black", back_color="white").save(OUT / "github_qr.png")
print("assets:", list(OUT.iterdir()))
