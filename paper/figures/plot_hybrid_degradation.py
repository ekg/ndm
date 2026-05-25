#!/usr/bin/env python3
"""
Generate Figure 4 (hybrid degradation) with per-seed points + SEM bars,
matching the styling of Figure 3 (plot_expressivity_seeds.py).

Data source: paper/results/figure_4_hybrid/canon_<pattern>__<task>__seed<seed>.json
mirrored from /home/erikg/elman/experiments/expressivity_tasks/results/
(see paper/results/figure_4_hybrid/SOURCES.md).

Three seeds (42, 123, 456) per (pattern, task). No mock / synthetic data.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "figure_4_hybrid"))

SEEDS = [42, 123, 456]
PATTERNS = ["pure_E88", "pure_FLA", "hybrid_AABB"]
PATTERN_LABEL = {
    "pure_E88":     "pure Emender",
    "pure_FLA":     "pure GDN",
    "hybrid_AABB":  "Emender+GDN\nhybrid (AABB)",
}
TASKS = ["modular_counter", "fsm_tracking"]
TASK_TITLE = {
    "modular_counter": "Modular counter ($K=5$)",
    "fsm_tracking":    "FSM tracking ($K=4$ states)",
}
TASK_RANDOM = {
    "modular_counter": 1 / 5,
    "fsm_tracking":    1 / 4,
}

# Colors matching plot_expressivity_seeds.py / paper convention:
#   Emender = blue, GDN = orange, hybrid = red (the standout in this fig).
COLORS = {
    "pure_E88":    "#4477AA",
    "pure_FLA":    "#EE7733",
    "hybrid_AABB": "#CC3311",
}


def load_seeds(pattern, task):
    accs = []
    for seed in SEEDS:
        path = os.path.join(
            DATA_DIR, f"canon_{pattern}__{task}__seed{seed}.json"
        )
        with open(path) as f:
            d = json.load(f)
        acc = d.get("final_acc", d.get("final_accuracy"))
        if acc is None:
            raise RuntimeError(f"no final_acc in {path}")
        accs.append(float(acc))
    return np.array(accs, dtype=float)


# Match the rcParams used by plot_expressivity_seeds.py for visual consistency.
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":  10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))

for ax, task in zip(axes, TASKS):
    n = len(PATTERNS)
    x = np.arange(n)
    width = 0.55

    for i, pattern in enumerate(PATTERNS):
        seeds = load_seeds(pattern, task)
        mean = seeds.mean()
        sem  = seeds.std(ddof=1) / np.sqrt(len(seeds))

        color = COLORS[pattern]

        # Bar (light fill, behind points)
        ax.bar(
            i, mean, width=width,
            color=color, alpha=0.25, zorder=1,
            edgecolor="none",
        )

        # SEM error bar
        ax.errorbar(
            i, mean, yerr=sem,
            fmt="none", color=color,
            capsize=4, capthick=1.2, linewidth=1.2,
            zorder=3,
        )

        # Individual seed points with horizontal jitter
        jitter = np.linspace(-0.08, 0.08, len(seeds))
        for s, jit in zip(seeds, jitter):
            ax.scatter(
                i + jit, s,
                color=color, s=32, zorder=4,
                edgecolors="white", linewidths=0.6,
            )

    # Random baseline
    rb = TASK_RANDOM[task]
    ax.axhline(
        rb, color="black", linestyle="--", linewidth=0.8,
        alpha=0.55, label=f"Random = {rb:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABEL[p] for p in PATTERNS], rotation=0, ha="center")
    ax.set_ylabel("Accuracy")
    ax.set_title(TASK_TITLE[task], pad=6)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend(loc="upper right", frameon=False)

fig.suptitle(
    "Hybrid degradation: Emender+GDN AABB underperforms either pure family "
    "(8 M scale, 3 seeds)",
    fontsize=9, y=1.02,
)
fig.tight_layout()

out = os.path.join(HERE, "hybrid_degradation_seeds.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
