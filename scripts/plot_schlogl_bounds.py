"""Plot Schlögl model thermodynamic-space concentration bounds (paper Fig. Schlogl).

Paper Eq. con_bound_Schlogl:
  exp(-(mu_X^0 - mu_B)) <= x <= exp(-(mu_X^0 - mu_A))
In log form (dimensionless /RT):
  -(mu_X^0 - mu_B) <= ln x <= -(mu_X^0 - mu_A)

We plot ln x bounds as a band versus control parameter A_e = mu_A - mu_B (driving).
We choose mu_B = 0, mu_A = A_e, and mu_X^0 = 0.
Then:
  0 <= ln x <= A_e

This reproduces the qualitative band used in the paper.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--Amax", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    A = np.linspace(0.0, args.Amax, args.n)
    lo = np.zeros_like(A)
    hi = A

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.fill_between(A, lo, hi, color="#b9e7b9", alpha=0.4, label="Thermodynamic space")
    ax.plot(A, lo, color="red", lw=2.3)
    ax.plot(A, hi, color="blue", lw=2.3)
    ax.set_xlabel(r"$A_{\mathbf{e}} = \mu_A-\mu_B$")
    ax.set_ylabel(r"$\ln x^{ss}$")
    ax.set_title("Schlögl: thermodynamic space bound")
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
