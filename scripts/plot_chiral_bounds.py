"""Plot chiral symmetry breaking thermodynamic bounds (paper Fig. chiral).

Paper Eq. SCSB_bound:
  -Δμ <= ln(r^{ss}/s^{ss}) <= +Δμ  (dimensionless /RT)

We plot ln(r/s) bounds as a band vs driving Δμ.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--Dmax", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    D = np.linspace(0.0, args.Dmax, args.n)
    lo = -D
    hi = D

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.fill_between(D, lo, hi, color="#b9e7b9", alpha=0.4, label="Thermodynamic space")
    ax.plot(D, lo, color="red", lw=2.3)
    ax.plot(D, hi, color="blue", lw=2.3)
    ax.set_xlabel(r"$\Delta\mu$")
    ax.set_ylabel(r"$\ln(r^{ss}/s^{ss})$")
    ax.set_title("Chiral symmetry breaking: bound")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")

    # Also save SVG
    svg_out = args.out.replace('.png', '.svg')
    fig.savefig(svg_out, format='svg', bbox_inches='tight')
    print(f"Wrote: {svg_out}")


if __name__ == "__main__":
    main()
