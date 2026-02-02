"""Self-assembly validation via *relaxation* (ODE integration).

Implements the correct workflow:
1) Fix thermodynamics: mu0_X, mu_Y (so Δμ is set)
2) Sample rates consistent with LDB (random prefactors only)
3) Fix conservation total M_tot, sample initial condition on that plane
4) Relax ODE to steady state
5) Plot points vs theoretical band (panel 1)

Usage:
  PYTHONPATH=code/crn python3 code/crn/scripts/sample_self_assembly_relaxation.py \
    --out papers/arxiv-2407.11498/notes/fig_self_assembly_relax_dots.png \
    --DeltaMu_over_RT 2.0 --n 300 --seed 0
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from crn_bounds.self_assembly import self_assembly_crn
from crn_bounds.ldb import sample_rates_from_ldb
from crn_bounds.model import relaxation, mass_action_flux_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--DeltaMu_over_RT", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_final", type=float, default=200.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dm = float(args.DeltaMu_over_RT)

    crn = self_assembly_crn()

    # Paper caption: mu1°=mu2°=mu3°=0 (dimensionless /RT)
    mu0_X = np.array([0.0, 0.0, 0.0])

    # External chemical potentials: choose mu_F - mu_W = Δμ.
    # Set mu_W=0, mu_F=dm.
    mu_Y = np.array([dm, 0.0])  # [F, W]

    # Chemostat concentrations can be 1; we encode driving via mu_Y in LDB sampler.
    y = np.array([1.0, 1.0])

    # Fixed conservation total
    M_tot = 10.0

    pts = []

    for _ in range(args.n):
        k_plus, k_minus = sample_rates_from_ldb(crn, mu0_X, mu_Y, rng=rng, loga_range=(-1.0, 1.0))

        # random initial condition on conservation plane
        x2 = np.exp(rng.uniform(-2.0, 1.0))
        x3 = np.exp(rng.uniform(-2.0, 1.0))
        x1 = M_tot - 2 * x2 - 3 * x3
        if x1 <= 1e-8:
            continue
        x0 = np.array([x1, x2, x3], dtype=float)

        try:
            x = relaxation(crn, k_plus, k_minus, y=y, x0=x0, t_final=args.t_final)
        except Exception:
            continue

        # Compute A1/RT for sanity (should be within [0, dm])
        _, Jp, Jm = mass_action_flux_split(x, crn, k_plus, k_minus, y=y)
        A1 = np.log(Jp[0] / Jm[0])

        pts.append((2 * np.log(x[0]), np.log(x[1]), A1))

    pts = np.array(pts)

    # Plot band: ln x2 in [ln x1^2, ln x1^2 + dm]
    if pts.size:
        x_lo, x_hi = np.quantile(pts[:, 0], [0.01, 0.99])
        pad = 0.25 * (x_hi - x_lo + 1e-9)
        x_grid = np.linspace(x_lo - pad, x_hi + pad, 300)
    else:
        x_grid = np.linspace(-2.0, 2.0, 300)

    y_lo = x_grid
    y_hi = x_grid + dm

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.fill_between(x_grid, y_lo, y_hi, color="#b9e7b9", alpha=0.35, label="Thermodynamic space")
    ax.plot(x_grid, y_lo, color="green", lw=2.5, label="Equilibrium")
    ax.plot(x_grid, y_hi, color="red", lw=2.5, ls="--", label="Amplification bound")

    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55)
        outside = (pts[:, 1] < pts[:, 0] - 1e-8) | (pts[:, 1] > pts[:, 0] + dm + 1e-8)
        if np.any(outside):
            ax.scatter(pts[outside, 0], pts[outside, 1], s=18, color="magenta", label="Outside")
            print('Outside count:', int(np.sum(outside)))
        # report A1 stats
        print('A1/RT min,max:', float(np.min(pts[:,2])), float(np.max(pts[:,2])))

    ax.set_xlabel(r"$\ln(x_1^2)$")
    ax.set_ylabel(r"$\ln(x_2)$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"Relaxation steady states (kept={len(pts)})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out} with {len(pts)} points")


if __name__ == "__main__":
    main()
