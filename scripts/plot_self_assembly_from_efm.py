"""Compute EFM for the self-assembly network and plot thermodynamic bounds.

This script demonstrates the *pipeline*:
  S (paper Eq. self_assembly_reac) -> split reversible -> EFM enumeration
  -> compute cycle affinity (DeltaMu) -> reproduce Fig self_assembly(e)-style bounds.

We keep the plotting identical to plot_self_assembly_A.py, but we source
DeltaMu from EFM cycle affinity.

Usage:
  PYTHONPATH=code/crn python3 code/crn/scripts/plot_self_assembly_from_efm.py \
    --out papers/arxiv-2407.11498/notes/fig_self_assembly_from_efm.png \
    --DeltaMu_over_RT 2.0 --mu1 0 --mu2 0 --mu3 0

Note: In the paper they set DeltaMu = mu_F - mu_W. Here we directly use the
input DeltaMu_over_RT to define A^Y for reaction 1 (F->W driven step).
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from crn_bounds.split import split_reversible
from crn_bounds.efm import enumerate_extreme_rays
from crn_bounds.stoichiometry import Stoichiometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--DeltaMu_over_RT", type=float, default=2.0)
    ap.add_argument("--mu1", type=float, default=0.0)
    ap.add_argument("--mu2", type=float, default=0.0)
    ap.add_argument("--mu3", type=float, default=0.0)
    args = ap.parse_args()

    dm = float(args.DeltaMu_over_RT)
    mu1, mu2, mu3 = float(args.mu1), float(args.mu2), float(args.mu3)

    # Paper stoichiometry matrix S (internal first 3 rows, then external F/W)
    # Columns correspond to reactions 1,2,3 in Eq. self_assembly_reac
    S_full = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
        [-1,  0,  0],
        [ 1,  0,  0],
    ], dtype=float)

    Sx = S_full[:3, :]  # internal block

    # Split reversible reactions into forward/backward irreversible
    split = split_reversible(Stoichiometry(Sx=Sx, Sy=None))
    Sx_split = split.stoich_split.Sx

    # Enumerate EFMs/extreme rays for Sx_split v = 0, v>=0
    rays = enumerate_extreme_rays(Sx_split)

    # Map rays back to original 3 reactions (net flux): v_orig[rho] = Σ sign[k]*v_split[k]
    efms_orig = []
    for r in rays:
        v = np.zeros(3)
        for k, (rho, sgn) in enumerate(zip(split.split_to_orig, split.split_sign)):
            v[rho] += sgn * r[k]
        efms_orig.append(v)

    # For this network, the only EFM direction should be proportional to [1,1,1].
    # We'll pick the first nonzero one.
    e = None
    for v in efms_orig:
        if np.linalg.norm(v) > 1e-12:
            e = v
            break
    if e is None:
        raise RuntimeError("No nonzero EFM found")

    # Normalize to make it easier to compare
    e = e / np.min(np.abs(e[np.abs(e) > 1e-12]))

    print("EFM in original reaction coordinates (should align with [1,1,1]):", e)

    # Cycle affinity A_e^Y = e^T A^Y.
    # In this self-assembly network, only reaction 1 has external driving: mu_F - mu_W = DeltaMu.
    # So we set A^Y = [DeltaMu, 0, 0] in units of RT (dimensionless) for plotting.
    A_Y = np.array([dm, 0.0, 0.0])
    DeltaMu_from_efm = float(e @ A_Y)
    if DeltaMu_from_efm < 0:
        DeltaMu_from_efm = -DeltaMu_from_efm

    # For e ~ [1,1,1], DeltaMu_from_efm ~ dm.
    dm_eff = DeltaMu_from_efm
    print("DeltaMu/RT from EFM cycle affinity:", dm_eff)

    # Plot exactly the same four panels as paper Fig self_assembly(e)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    # Panel 1: ln(x2) vs ln(x1^2)
    x = np.linspace(-2.0, 2.0, 200)
    y_eq = x - (mu2 - 2 * mu1)
    y_amp = y_eq + dm_eff
    ax = axes[0]
    ax.fill_between(x, y_eq, y_amp, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq, color="green", lw=2, label="Equilibrium")
    ax.plot(x, y_amp, color="red", lw=2, ls="--", label="Amplification bound")
    ax.set_xlabel(r"$\ln(x_1^2)$")
    ax.set_ylabel(r"$\ln(x_2)$")

    # Panel 2: ln(x3) vs ln(x1 x2)
    x = np.linspace(-2.0, 3.0, 200)
    y_amp_low = x - (mu3 - mu2 - mu1 + dm_eff)
    y_eq_high = x - (mu3 - mu2 - mu1)
    ax = axes[1]
    ax.fill_between(x, y_amp_low, y_eq_high, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq_high, color="green", lw=2)
    ax.plot(x, y_amp_low, color="red", lw=2, ls="--")
    ax.set_xlabel(r"$\ln(x_1 x_2)$")
    ax.set_ylabel(r"$\ln(x_3)$")

    # Panel 3: ln(x1^3) vs ln(x3)
    x = np.linspace(-2.0, 2.5, 200)
    y_amp_low = x - (mu3 - 3 * mu1 + dm_eff)
    y_eq_high = x - (mu3 - 3 * mu1)
    ax = axes[2]
    ax.fill_between(x, y_amp_low, y_eq_high, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq_high, color="green", lw=2)
    ax.plot(x, y_amp_low, color="red", lw=2, ls="--")
    ax.set_xlabel(r"$\ln(x_3)$")
    ax.set_ylabel(r"$\ln(x_1^3)$")

    # Panel 4: ln(x3^2) vs ln(x2^3)
    # From Eq self_assembly_probe_bound: dm appears with factor 3.
    x = np.linspace(-1.0, 8.0, 200)
    y_amp_low = x - (2 * mu3 - 3 * mu2 + 3 * dm_eff)
    y_eq_high = x - (2 * mu3 - 3 * mu2)
    ax = axes[3]
    ax.fill_between(x, y_amp_low, y_eq_high, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq_high, color="green", lw=2)
    ax.plot(x, y_amp_low, color="red", lw=2, ls="--")
    ax.set_xlabel(r"$\ln(x_2^3)$")
    ax.set_ylabel(r"$\ln(x_3^2)$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle("Self-assembly bounds from EFM enumeration (paper network)")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")

    # Also save SVG
    svg_out = args.out.replace('.png', '.svg')
    fig.savefig(svg_out, format='svg', bbox_inches='tight')
    print(f"Wrote: {svg_out}")


if __name__ == "__main__":
    main()
