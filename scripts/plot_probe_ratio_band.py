"""Plot a 2D band for a probe log-ratio bound.

Given a probe stoichiometry s and bounds on y = s^T ln x in [lo,hi],
we can visualize the bound for a chosen pair of axes (u, v) where u and v
are linear forms in ln x (e.g. u=ln(x2^3), v=ln(x3^2)).

This script is mainly for the paper-style visualization of probe bounds.

Usage (self-assembly probe example):
  PYTHONPATH=code/crn python3 code/crn/scripts/plot_probe_ratio_band.py \
    --out papers/arxiv-2407.11498/notes/fig_probe_ratio_band.png \
    --x_label "ln(x2^3)" --y_label "ln(x3^2)" \
    --lo -6 --hi 0
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--x_label", type=str, default="x")
    ap.add_argument("--y_label", type=str, default="y")
    ap.add_argument("--x_min", type=float, default=-1.0)
    ap.add_argument("--x_max", type=float, default=8.0)
    args = ap.parse_args()

    lo, hi = args.lo, args.hi

    x = np.linspace(args.x_min, args.x_max, 250)
    y_lo = x + lo
    y_hi = x + hi

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.fill_between(x, y_lo, y_hi, color="#b9e7b9", alpha=0.5, label="Thermodynamic space")
    ax.plot(x, y_hi, color="green", lw=2.2, label="Upper bound")
    ax.plot(x, y_lo, color="red", lw=2.2, ls="--", label="Lower bound")
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
