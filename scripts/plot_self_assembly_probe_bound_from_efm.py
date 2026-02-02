"""Compute self-assembly probe bound from enumerated EFMs and plot.

Goal: derive Eq. self_assembly_probe_bound *from EFMs* of extended network.

We follow the paper logic:
- Extended network has 4 reactions: rho1,rho2,rho3,probe(rho4)
- Enumerate EFMs (in original coordinates) e_hat_k
- Each EFM corresponds to a pathway converting between probe sides.

For the probe interconversion 3 X2 <-> 2 X3, define log ratio:
  y = ln((x3^2)/(x2^3))
Paper gives:
  -(2mu3-3mu2+3Δμ)/RT <= y <= -(2mu3-3mu2)/RT

We reconstruct the Δμ coefficient from which EFMs include reaction 1.
In this model only reaction 1 carries external affinity Δμ.

Usage:
  PYTHONPATH=code/crn python3 code/crn/scripts/plot_self_assembly_probe_bound_from_efm.py \
    --out papers/arxiv-2407.11498/notes/fig_self_assembly_probe_from_efm.png \
    --DeltaMu_over_RT 2.0 --mu2 0 --mu3 0
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from crn_bounds.stoichiometry import Stoichiometry
from crn_bounds.split import split_reversible
from crn_bounds.efm import enumerate_extreme_rays


def _norm_dir(v: np.ndarray) -> np.ndarray:
    nz = np.abs(v[np.abs(v) > 1e-10])
    if nz.size == 0:
        return v
    v = v / np.min(nz)
    k = np.argmax(np.abs(v) > 1e-10)
    if v[k] < 0:
        v = -v
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--DeltaMu_over_RT", type=float, default=2.0)
    ap.add_argument("--mu2", type=float, default=0.0)
    ap.add_argument("--mu3", type=float, default=0.0)
    args = ap.parse_args()

    dm = float(args.DeltaMu_over_RT)
    mu2, mu3 = float(args.mu2), float(args.mu3)

    # Extended S (X1,X2,X3,F,W plus probe col)
    S_full = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
        [-1,  0,  0],
        [ 1,  0,  0],
    ], dtype=float)
    probe = np.array([[0], [-3], [2], [0], [0]], dtype=float)
    S_ext = np.hstack([S_full, probe])
    Sx_ext = S_ext[:3, :]

    split = split_reversible(Stoichiometry(Sx=Sx_ext, Sy=None))
    rays = enumerate_extreme_rays(split.stoich_split.Sx)

    # map back to original 4 reactions
    efms = []
    for r in rays:
        v = np.zeros(4)
        for k, (rho, sgn) in enumerate(zip(split.split_to_orig, split.split_sign)):
            v[rho] += sgn * r[k]
        if np.linalg.norm(v) > 1e-10:
            efms.append(_norm_dir(v))

    # external affinity vector (dimensionless /RT): only reaction 1 has dm
    A_Y = np.array([dm, 0.0, 0.0, 0.0])

    # For each EFM, pathway external affinity = e^T A_Y
    # For probe conversion, we want extremes for y = ln(x3^2/x2^3)
    # which corresponds to stoich s = [0,-3,2] (only depends on mu2,mu3 and dm)
    # paper says lower uses +3 dm, upper uses 0 dm.

    vals = [float(e @ A_Y) for e in efms]
    # Only EFMs that actually include probe (component 4 nonzero): they all should.
    probe_vals = [v for e, v in zip(efms, vals) if abs(e[3]) > 1e-10]

    # The most dissipative pathway for dimer->trimer has coefficient 3*dm in paper;
    # Here we just take max external affinity among EFMs and scale by probe participation (e4=1)
    max_ext = max(probe_vals)
    min_ext = min(probe_vals)

    # Convert to y-bounds (dimensionless, divide by RT already):
    # y in [-(2mu3-3mu2 + max_ext)/RT,  -(2mu3-3mu2 + min_ext)/RT]
    # Note: mu are already /RT inputs here, so keep consistent.
    base = (2 * mu3 - 3 * mu2)
    y_lo = -(base + max_ext)
    y_hi = -(base + min_ext)

    # Plot as band in (ln x2^3, ln x3^2)
    x = np.linspace(-1.0, 8.0, 200)
    y_eq = x - base  # equilibrium boundary (no driving)
    # driven lower boundary
    y_amp = x + y_lo  # since y = ln(x3^2) - ln(x2^3)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    # feasible region between amplification (lower) and equilibrium (upper)
    ax.fill_between(x, y_amp, y_eq, color="#b9e7b9", alpha=0.6)
    ax.plot(x, y_eq, color="green", lw=2, label="Equilibrium")
    ax.plot(x, y_amp, color="red", lw=2, ls="--", label="Amplification bound (from EFMs)")
    ax.set_xlabel(r"$\ln(x_2^3)$")
    ax.set_ylabel(r"$\ln(x_3^2)$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Probe bound for 3X2 <-> 2X3 (from EFMs)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    # Also save SVG
    svg_out = args.out.replace('.png', '.svg')
    fig.savefig(svg_out, format='svg', bbox_inches='tight')
    print(f"Wrote: {svg_out}")

    print("EFM external affinities (e^T A_Y):", probe_vals)
    print("Derived y bounds:", y_lo, y_hi)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
