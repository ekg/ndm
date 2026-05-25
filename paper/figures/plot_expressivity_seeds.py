#!/usr/bin/env python3
"""
Generate §6 expressivity bar charts with per-seed points + SEM error bars.

Data source: paper/ndmpapernotes.md lines 153-173
Three seeds per condition; seed values recovered from (mean, min, max):
  seed_min = min, seed_max = max, seed_middle = 3*mean - min - max
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data (mean, min, max from 3 seeds) ────────────────────────────────────────
# Source: paper/ndmpapernotes.md, "Current matched 8M S3/S5 run, three seeds"

data_s5 = {
    "Emender":      {"mean": 0.7918, "min": 0.6232, "max": 0.8880},
    "GDN":          {"mean": 0.3552, "min": 0.3148, "max": 0.3850},
    "M²RNN-CMA":   {"mean": 0.2157, "min": 0.1856, "max": 0.2309},
    "M²RNN-paper": {"mean": 0.1698, "min": 0.1555, "max": 0.1844},
}

data_s3 = {
    "Emender":      {"mean": 1.0000, "min": 0.9999, "max": 1.0000},
    "GDN":          {"mean": 0.7185, "min": 0.6122, "max": 0.8516},
    "M²RNN-CMA":   {"mean": 0.3124, "min": 0.2742, "max": 0.3529},
    "M²RNN-paper": {"mean": 0.3773, "min": 0.3669, "max": 0.3855},
}

def recover_seeds(d):
    """Recover three seed values from (mean, min, max) of 3 seeds."""
    mn, mx, mu = d["min"], d["max"], d["mean"]
    middle = 3 * mu - mn - mx
    # Clamp to [0, 1] due to floating-point rounding
    middle = max(mn, min(mx, middle))
    return sorted([mn, middle, mx])

COLORS = {
    "Emender":      "#4477AA",   # blue
    "GDN":          "#EE7733",   # orange
    "M²RNN-CMA":   "#CC3311",   # red
    "M²RNN-paper": "#999933",   # olive
}
RANDOM_S5 = 1 / 120  # 0.0083
RANDOM_S3 = 1 / 6    # 0.1667

# ── Style matching Figure 2 ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))

for ax, data, task, random_bl, title in [
    (axes[0], data_s5, "S5", RANDOM_S5, "$S_5$ accuracy at training length $T=128$"),
    (axes[1], data_s3, "S3", RANDOM_S3, "$S_3$ accuracy at training length $T=128$"),
]:
    models = list(data.keys())
    n = len(models)
    x = np.arange(n)
    width = 0.55

    for i, model in enumerate(models):
        seeds = recover_seeds(data[model])
        mean = data[model]["mean"]
        sem = np.std(seeds, ddof=1) / np.sqrt(3)

        color = COLORS[model]
        light_color = color + "55"  # semi-transparent

        # Bar (light, behind points)
        ax.bar(
            i, mean, width=width,
            color=color, alpha=0.25, zorder=1,
            edgecolor="none"
        )

        # SEM error bar
        ax.errorbar(
            i, mean, yerr=sem,
            fmt="none", color=color,
            capsize=4, capthick=1.2, linewidth=1.2,
            zorder=3
        )

        # Individual seed points
        jitter = np.linspace(-0.08, 0.08, 3)
        for j, (s, jit) in enumerate(zip(seeds, jitter)):
            ax.scatter(
                i + jit, s,
                color=color, s=32, zorder=4,
                edgecolors="white", linewidths=0.6
            )

    # Random baseline
    ax.axhline(random_bl, color="black", linestyle="--", linewidth=0.8,
               alpha=0.55, label=f"Random = {random_bl:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, pad=6)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend(loc="upper right", frameon=False)

fig.suptitle(
    "Expressivity separation: permutation-composition probes (8 M scale, 3 seeds)",
    fontsize=9, y=1.01
)
fig.tight_layout()

out = "paper/figures/s5_expressivity_seeds.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
