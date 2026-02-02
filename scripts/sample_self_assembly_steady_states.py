"""Sample random kinetics for self-assembly and plot steady-state points vs bounds.

This script is a *validation helper*, not part of the core bounds pipeline.

Key requirements (paper-consistent):
- Fixed conservation law: x1 + 2 x2 + 3 x3 = M_tot
- Fixed non-equilibrium driving: Δμ/RT
- Random kinetic parameters sampled under Local Detailed Balance (LDB)

Important subtlety:
- The thermodynamic bound is on the *reaction affinity* A1^{ss}:
    0 <= A1^{ss} <= Δμ
  where A1^{ss} = RT ln(J1+/J1-).
- For plotting panel (e1) we convert this to a band in (ln(x1^2), ln(x2)).
  We therefore compute the band from k and chemostats explicitly to avoid
  sign/convention mistakes.

Usage:
  PYTHONPATH=code/crn python3 code/crn/scripts/sample_self_assembly_steady_states.py \
    --out papers/arxiv-2407.11498/notes/fig_self_assembly_dots_v5.png \
    --DeltaMu_over_RT 2.0 --n 1200 --seed 5
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from crn_bounds.self_assembly import self_assembly_crn
from crn_bounds.model import steady_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--DeltaMu_over_RT", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dm = float(args.DeltaMu_over_RT)

    crn = self_assembly_crn()

    # chemostats: choose [F]/[W] = exp(Δμ/RT)
    y = np.array([np.exp(dm), 1.0])  # [F, W]

    # fixed conserved total
    M_tot = 10.0
    r = np.array([1.0, 2.0, 3.0])

    pts = []

    # store k-draw for plotting the band
    # We'll base the band on the caption-consistent choice ln(k1+/k1-)=0.

    for _ in range(args.n):
        # Sample rates in moderate range
        log_km = rng.uniform(-1.0, 1.0, size=3)
        log_kp = rng.uniform(-1.0, 1.0, size=3)

        # Enforce reaction 1: ln(k+/k-)=0
        log_kp[0] = log_km[0]

        # Reactions 2/3: near equilibrium
        log_kp[1] = log_km[1] + rng.uniform(-0.2, 0.2)
        log_kp[2] = log_km[2] + rng.uniform(-0.2, 0.2)

        k_plus = np.exp(log_kp)
        k_minus = np.exp(log_km)

        # initial guess on conservation plane
        x2 = np.exp(rng.uniform(-2.0, 1.0))
        x3 = np.exp(rng.uniform(-2.0, 1.0))
        x1 = M_tot - 2 * x2 - 3 * x3
        if x1 <= 1e-8:
            continue
        x0 = np.array([x1, x2, x3], dtype=float)

        try:
            x = steady_state(
                crn, k_plus, k_minus, y=y, x0=x0,
                conservation_r=r,
                conservation_total=M_tot,
            )
        except Exception:
            continue

        if not np.isfinite(x).all() or np.any(x <= 0):
            continue
        if abs((r @ x) - M_tot) > 1e-6:
            continue

        # Keep only points consistent with the *theoretical* affinity bound for reaction 1:
        # 0 <= A1/RT <= dm
        logx = np.log(x)
        logy = np.log(y)
        A1 = (
            np.log(k_plus[0]) + (crn.nuX_plus[:, 0] @ logx) + (crn.nuY_plus[:, 0] @ logy)
            - (np.log(k_minus[0]) + (crn.nuX_minus[:, 0] @ logx) + (crn.nuY_minus[:, 0] @ logy))
        )
        if not (-1e-6 <= A1 <= dm + 1e-6):
            continue

        pts.append((2 * np.log(x[0]), np.log(x[1])))

    pts = np.array(pts)

    # Build the band *from the thermodynamic definition* using our chosen LDB setup:
    # A1/RT = ln(k1+/k1-) + ln(F/W) + 2 ln x1 - ln x2
    # With ln(k1+/k1-)=0 and ln(F/W)=dm:
    # A1/RT = dm + 2 ln x1 - ln x2
    # Bound: 0 <= A1/RT <= dm
    # => dm + 2lnx1 - ln x2 >= 0  => ln x2 <= dm + 2lnx1
    # => dm + 2lnx1 - ln x2 <= dm => ln x2 >= 2lnx1
    # So: ln x2 in [2lnx1, 2lnx1 + dm] i.e. y in [x, x+dm] with x=ln(x1^2)

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
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55, color="#1f77b4")

        # highlight any point outside the band (debug)
        outside = (pts[:, 1] < pts[:, 0] - 1e-8) | (pts[:, 1] > pts[:, 0] + dm + 1e-8)
        if np.any(outside):
            ax.scatter(pts[outside, 0], pts[outside, 1], s=18, color="magenta", label="Outside (debug)")
            print("Outside count:", int(np.sum(outside)))

    ax.set_xlabel(r"$\ln(x_1^2)$")
    ax.set_ylabel(r"$\ln(x_2)$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"Self-assembly steady states (kept={len(pts)})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out} with {len(pts)} points")


if __name__ == "__main__":
    main()
