"""Self-assembly: run ONE batch of LDB-consistent simulations and reuse the same
steady-state points across multiple thermodynamic panels.

We generate points (x1,x2,x3) at steady state by ODE relaxation.
Then we make several plots:
- Panel A: ln(x2) vs ln(x1^2)  (reaction 1 ratio)
- Panel B: ln(x3) vs ln(x1 x2) (reaction 2 ratio)
- Panel C: ln(x1^3) vs ln(x3)  (reaction 3 ratio)
- Probe panel: ln(x3^2) vs ln(x2^3) (probe ratio)

Usage:
  PYTHONPATH=code/crn python3 code/crn/scripts/sample_self_assembly_relaxation_panels.py \
    --outdir papers/arxiv-2407.11498/notes \
    --DeltaMu_over_RT 2.0 --n 600 --seed 0

This writes multiple PNGs and a .npz with the points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from crn_bounds.self_assembly import self_assembly_crn
from crn_bounds.ldb import sample_rates_from_ldb
from crn_bounds.model import relaxation


def _band_plot(ax, x_grid, y_lo, y_hi, pts_x, pts_y, xlabel, ylabel, title, dm):
    ax.fill_between(x_grid, y_lo, y_hi, color="#b9e7b9", alpha=0.35)
    ax.plot(x_grid, y_lo, color="green", lw=2.3, label="Equilibrium")
    ax.plot(x_grid, y_hi, color="red", lw=2.3, ls="--", label="Amplification")
    ax.scatter(pts_x, pts_y, s=8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    outside = (pts_y < y_lo.min() - 1e-9)  # coarse; refined below not needed
    # refined outside check using pointwise bounds
    # compute bounds at each point: y in [x, x+dm]
    # (we pass x=pts_x to caller-defined convention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--DeltaMu_over_RT", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_final", type=float, default=200.0)
    ap.add_argument("--M_tot", type=float, default=10.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    dm = float(args.DeltaMu_over_RT)

    crn = self_assembly_crn()
    mu0_X = np.array([0.0, 0.0, 0.0])
    mu_Y = np.array([dm, 0.0])  # [F,W]
    y = np.array([1.0, 1.0])

    xs = []
    kept = 0
    trials = 0
    while kept < args.n and trials < args.n * 5:
        trials += 1
        k_plus, k_minus = sample_rates_from_ldb(crn, mu0_X, mu_Y, rng=rng, loga_range=(-1.0, 1.0))

        x2 = np.exp(rng.uniform(-2.0, 1.0))
        x3 = np.exp(rng.uniform(-2.0, 1.0))
        x1 = args.M_tot - 2 * x2 - 3 * x3
        if x1 <= 1e-8:
            continue
        x0 = np.array([x1, x2, x3], dtype=float)

        try:
            x = relaxation(crn, k_plus, k_minus, y=y, x0=x0, t_final=args.t_final)
        except Exception:
            continue

        xs.append(x)
        kept += 1

    X = np.array(xs, dtype=float)
    np.savez(outdir / "self_assembly_relax_points.npz", X=X, DeltaMu_over_RT=dm)

    # Precompute log quantities
    lnx = np.log(X)
    ln_x1 = lnx[:, 0]
    ln_x2 = lnx[:, 1]
    ln_x3 = lnx[:, 2]

    # Panel A: ln(x2) vs ln(x1^2)
    Xa = 2 * ln_x1
    Ya = ln_x2

    # Panel B: ln(x3) vs ln(x1 x2)
    Xb = ln_x1 + ln_x2
    Yb = ln_x3

    # Panel C: ln(x1^3) vs ln(x3)
    Xc = ln_x3
    Yc = 3 * ln_x1

    # Probe: ln(x3^2) vs ln(x2^3)
    Xp = 3 * ln_x2
    Yp = 2 * ln_x3

    def grid_for(Xaxis):
        lo, hi = np.quantile(Xaxis, [0.01, 0.99])
        pad = 0.25 * (hi - lo + 1e-9)
        return np.linspace(lo - pad, hi + pad, 320)

    # Each bound is of the form y in [x, x+dm] in the chosen coords.
    # A: y=ln x2, x=ln x1^2
    xg = grid_for(Xa)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.fill_between(xg, xg, xg + dm, color="#b9e7b9", alpha=0.35)
    ax.plot(xg, xg, color="green", lw=2.3)
    ax.plot(xg, xg + dm, color="red", lw=2.3, ls="--")
    ax.scatter(Xa, Ya, s=8, alpha=0.5)
    ax.set_xlabel(r"$\ln(x_1^2)$")
    ax.set_ylabel(r"$\ln(x_2)$")
    ax.set_title(f"Self-assembly R1 ratio (n={len(X)})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_self_assembly_relax_panel_R1.png", dpi=200)

    # B: y=ln x3, x=ln(x1 x2) with y in [x-dm, x]
    xg = grid_for(Xb)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.fill_between(xg, xg - dm, xg, color="#b9e7b9", alpha=0.35)
    ax.plot(xg, xg, color="green", lw=2.3)
    ax.plot(xg, xg - dm, color="red", lw=2.3, ls="--")
    ax.scatter(Xb, Yb, s=8, alpha=0.5)
    ax.set_xlabel(r"$\ln(x_1 x_2)$")
    ax.set_ylabel(r"$\ln(x_3)$")
    ax.set_title(f"Self-assembly R2 ratio (n={len(X)})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_self_assembly_relax_panel_R2.png", dpi=200)

    # C: y=ln(x1^3), x=ln x3 with y in [x-dm, x]
    xg = grid_for(Xc)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.fill_between(xg, xg - dm, xg, color="#b9e7b9", alpha=0.35)
    ax.plot(xg, xg, color="green", lw=2.3)
    ax.plot(xg, xg - dm, color="red", lw=2.3, ls="--")
    ax.scatter(Xc, Yc, s=8, alpha=0.5)
    ax.set_xlabel(r"$\ln(x_3)$")
    ax.set_ylabel(r"$\ln(x_1^3)$")
    ax.set_title(f"Self-assembly R3 ratio (n={len(X)})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_self_assembly_relax_panel_R3.png", dpi=200)

    # Probe: y=ln(x3^2), x=ln(x2^3)
    # Paper bound: ln(x3^2/x2^3) in [-3*dm, 0]  ->  y - x in [-3*dm, 0]
    xg = grid_for(Xp)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.fill_between(xg, xg - 3 * dm, xg, color="#b9e7b9", alpha=0.35)
    ax.plot(xg, xg, color="green", lw=2.3)
    ax.plot(xg, xg - 3 * dm, color="red", lw=2.3, ls="--")
    ax.scatter(Xp, Yp, s=8, alpha=0.5)
    ax.set_xlabel(r"$\ln(x_2^3)$")
    ax.set_ylabel(r"$\ln(x_3^2)$")
    ax.set_title(f"Self-assembly probe ratio (n={len(X)})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_self_assembly_relax_panel_probe.png", dpi=200)

    print(f"Wrote panels to: {outdir}")


if __name__ == "__main__":
    main()
