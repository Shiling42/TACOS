"""Plot A (as in paper Fig. self_assembly(e)) for the self-assembly network.

We reproduce the *bounds* (green equilibrium line + red amplification bound)
using the explicit inequalities given in main.tex:
  - Eq. (self_assembly_bound)
  - Eq. (self_assembly_probe_bound)

We do NOT simulate kinetics here (dots in the paper), only plot the bounds.

Usage:
  PYTHONPATH=code/crn python3 code/crn/scripts/plot_self_assembly_A.py \
    --out papers/arxiv-2407.11498/notes/fig_self_assembly_A_repro.png \
    --DeltaMu_over_RT 2.0 \
    --mu1 0 --mu2 0 --mu3 0
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--DeltaMu_over_RT", type=float, default=2.0)
    ap.add_argument("--mu1", type=float, default=0.0)
    ap.add_argument("--mu2", type=float, default=0.0)
    ap.add_argument("--mu3", type=float, default=0.0)
    args = ap.parse_args()

    dm = args.DeltaMu_over_RT
    mu1, mu2, mu3 = args.mu1, args.mu2, args.mu3

    # We'll plot the four panels corresponding to ratios in the paper.
    # Use broad x-ranges in log-space.

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    # Panel 1: ln(x2) vs ln(x1^2)  i.e. y=ln x2 , x=ln(x1^2)=2 ln x1
    # Bounds from Eq self_assembly_bound (first line):
    #   exp(-(mu2-2mu1)) <= x2/x1^2 <= exp(-(mu2-2mu1-dm))
    # => ln x2 between 2 ln x1 -(mu2-2mu1)  and 2 ln x1 -(mu2-2mu1-dm)
    x = np.linspace(-2.0, 2.0, 200)  # x = ln(x1^2)
    # equilibrium line (green): ln x2 = x - (mu2-2mu1)
    y_eq = x - (mu2 - 2 * mu1)
    # amplification bound (red dashed): ln x2 = x - (mu2-2mu1-dm) = y_eq + dm
    y_amp = y_eq + dm

    ax = axes[0]
    ax.fill_between(x, y_eq, y_amp, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq, color="green", lw=2, label="Equilibrium")
    ax.plot(x, y_amp, color="red", lw=2, ls="--", label="Amplification bound")
    ax.set_xlabel(r"$\ln(x_1^2)$")
    ax.set_ylabel(r"$\ln(x_2)$")

    # Panel 2: ln(x3) vs ln(x1 x2)
    # Eq self_assembly_bound (second line):
    #   exp(-(mu3-mu2-mu1+dm)) <= x3/(x1 x2) <= exp(-(mu3-mu2-mu1))
    # => ln x3 between ln(x1 x2) -(mu3-mu2-mu1+dm) and ln(x1 x2) -(mu3-mu2-mu1)
    x = np.linspace(-2.0, 3.0, 200)  # x = ln(x1 x2)
    y_amp_low = x - (mu3 - mu2 - mu1 + dm)  # lower boundary (amplification, red dashed)
    y_eq_high = x - (mu3 - mu2 - mu1)       # upper boundary (equilibrium, green)

    ax = axes[1]
    ax.fill_between(x, y_amp_low, y_eq_high, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq_high, color="green", lw=2, label="Equilibrium")
    ax.plot(x, y_amp_low, color="red", lw=2, ls="--", label="Amplification bound")
    ax.set_xlabel(r"$\ln(x_1 x_2)$")
    ax.set_ylabel(r"$\ln(x_3)$")

    # Panel 3: ln(x1^3) vs ln(x3)
    # Eq self_assembly_bound (third line):
    #   exp(-(mu3-3mu1+dm)) <= x1^3/x3 <= exp(-(mu3-3mu1)
    # => ln(x1^3) between ln(x3) -(mu3-3mu1+dm) and ln(x3) -(mu3-3mu1)
    x = np.linspace(-2.0, 2.5, 200)  # x = ln(x3)
    y_amp_low = x - (mu3 - 3 * mu1 + dm)
    y_eq_high = x - (mu3 - 3 * mu1)

    ax = axes[2]
    ax.fill_between(x, y_amp_low, y_eq_high, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq_high, color="green", lw=2, label="Equilibrium")
    ax.plot(x, y_amp_low, color="red", lw=2, ls="--", label="Amplification bound")
    ax.set_xlabel(r"$\ln(x_3)$")
    ax.set_ylabel(r"$\ln(x_1^3)$")

    # Panel 4: ln(x3^2) vs ln(x2^3)
    # Eq self_assembly_probe_bound:
    #   exp(-(2mu3-3mu2+3dm)) <= (x3^2)/(x2^3) <= exp(-(2mu3-3mu2))
    # => ln(x3^2) between ln(x2^3) -(2mu3-3mu2+3dm) and ln(x2^3) -(2mu3-3mu2)
    x = np.linspace(-1.0, 8.0, 200)  # x = ln(x2^3)
    y_amp_low = x - (2 * mu3 - 3 * mu2 + 3 * dm)
    y_eq_high = x - (2 * mu3 - 3 * mu2)

    ax = axes[3]
    ax.fill_between(x, y_amp_low, y_eq_high, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq_high, color="green", lw=2, label="Equilibrium")
    ax.plot(x, y_amp_low, color="red", lw=2, ls="--", label="Amplification bound")
    ax.set_xlabel(r"$\ln(x_2^3)$")
    ax.set_ylabel(r"$\ln(x_3^2)$")

    # One legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle(r"Self-assembly thermodynamic space bounds (from Eqs. self\_assembly\_bound + self\_assembly\_probe\_bound)")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
