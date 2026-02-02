"""Plot thermodynamic space bounds (Plot A: per-species interval).

Usage:
  PYTHONPATH=code/crn python code/crn/scripts/plot_thermo_space_A.py \
    --out papers/arxiv-2407.11498/notes/fig_thermo_space_A.png

This uses the same toy Fig.3 setup as the unit test.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from crn_bounds.thermodynamics import visualize_thermo_space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Fig.3 toy network (same as test_paper_fig3)
    S = np.array([
        [-1,  0,  1,  0],  # X1
        [ 1, -1,  0,  1],  # X2
        [ 0,  1, -1,  0],  # X3
        [ 0,  0,  0, -1],  # X4
    ], dtype=float)

    dG_0 = np.array([0.0, 2.0, 1.0, -1.0], dtype=float) * 1000.0  # J/mol
    dG_Y = np.array([-0.5, -0.5, 1.0, -0.5], dtype=float) * 1000.0  # J/mol

    E = [
        np.array([1, 1, 1, 0], dtype=float),
        np.array([0, -1, -1, 1], dtype=float),
        np.array([1, 0, -1, 1], dtype=float),
        np.array([-1, 1, 1, -1], dtype=float),
    ]

    x_min, x_max, names = visualize_thermo_space(
        S, dG_0, dG_Y, E,
        species_names=["X1", "X2", "X3", "X4"],
    )

    # Prepare plot
    xs = np.arange(len(names))

    # Replace inf upper bounds with a finite cap for plotting (log scale)
    finite_max = x_max[np.isfinite(x_max)]
    cap = float(np.max(finite_max)) if finite_max.size else 1.0
    cap *= 1e2  # show "open" intervals as reaching to cap*100

    y_hi_plot = np.where(np.isfinite(x_max), x_max, cap)

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (lo, hi) in enumerate(zip(x_min, y_hi_plot)):
        ax.plot([i, i], [lo, hi], color="tab:green", linewidth=4)
        ax.scatter([i, i], [lo, hi], color="black", s=18, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("concentration x (arb. units)")
    ax.set_title("Thermodynamic Space (Plot A): per-species bounds")

    # annotate inf
    for i, hi in enumerate(x_max):
        if not np.isfinite(hi):
            ax.text(i, cap, "∞", ha="center", va="bottom")

    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
