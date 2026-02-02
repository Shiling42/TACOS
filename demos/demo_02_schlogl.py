#!/usr/bin/env python3
"""
TACOS Demo 2: Schlögl Model Bifurcation Diagram
================================================

Network:
  R1: 2X + A ⇌ 3X   (autocatalytic)
  R2: X ⇌ B         (degradation)

Parameters from paper: k1± = 1, k2± = 8, b = 0.02, vary a

Thermodynamic bounds (Eq. 23):
  With μ°_X = μ°_A = μ°_B = 0:
  ln(b) ≤ ln(x^ss) ≤ ln(a)

  In terms of cycle affinity A_e/RT = ln(a/b):
  ln(b) ≤ ln(x^ss) ≤ ln(b) + A_e/RT

Reference: arXiv:2407.11498, Fig. Schlogl
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    'bound_fill': '#b9e7b9',
    'upper_bound': 'blue',
    'lower_bound': 'red',
    'stable': '#2c3e50',
    'unstable': '#95a5a6',
}


def find_steady_states(a, b=0.02, k1p=1.0, k1m=1.0, k2p=8.0, k2m=8.0):
    """Find steady states by solving the cubic analytically."""
    # dx/dt = k1p*a*x² - k1m*x³ - k2p*x + k2m*b = 0
    # => k1m*x³ - k1p*a*x² + k2p*x - k2m*b = 0
    coeffs = [k1m, -k1p * a, k2p, -k2m * b]
    roots = np.roots(coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_roots.append(r.real)
    return sorted(real_roots)


def check_stability(x_ss, a, k1p=1.0, k1m=1.0, k2p=8.0):
    """Stable if df/dx < 0 at steady state."""
    # f(x) = k1p*a*x² - k1m*x³ - k2p*x + k2m*b
    # df/dx = 2*k1p*a*x - 3*k1m*x² - k2p
    dfdx = 2 * k1p * a * x_ss - 3 * k1m * x_ss**2 - k2p
    return dfdx < 0


def compute_bifurcation(A_e_values, b=0.02):
    """Compute bifurcation diagram: steady states vs A_e."""
    results = []
    for A_e in A_e_values:
        a = b * np.exp(A_e)  # A_e/RT = ln(a/b) => a = b*exp(A_e)
        roots = find_steady_states(a, b)
        for x_ss in roots:
            stable = check_stability(x_ss, a)
            results.append({
                'A_e': A_e,
                'a': a,
                'x_ss': x_ss,
                'ln_x': np.log(x_ss),
                'stable': stable,
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='notes/demo_schlogl.png')
    parser.add_argument('--svg', action='store_true', help='Also save SVG')
    args = parser.parse_args()

    outdir = Path(args.out).parent
    outdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("TACOS Demo 2: Schlögl Model Bifurcation")
    print("=" * 60)
    print("\nNetwork: R1: 2X + A ⇌ 3X,  R2: X ⇌ B")
    print("Parameters: k1± = 1, k2± = 8, b = 0.02")

    b = 0.02

    # Compute bifurcation - need A_e > 5.5 for bistability
    A_e_range = np.linspace(0, 8, 400)
    bif_data = compute_bifurcation(A_e_range, b)

    stable = [d for d in bif_data if d['stable']]
    unstable = [d for d in bif_data if not d['stable']]
    print(f"\nFound {len(stable)} stable, {len(unstable)} unstable points")

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Thermodynamic bounds
    A_e_plot = np.linspace(0, 8, 200)
    ln_b = np.log(b)
    lo_bound = np.full_like(A_e_plot, ln_b)
    hi_bound = ln_b + A_e_plot

    ax.fill_between(A_e_plot, lo_bound, hi_bound, color=COLORS['bound_fill'],
                    alpha=0.4, label='Thermodynamic space')
    ax.plot(A_e_plot, lo_bound, color=COLORS['lower_bound'], lw=2.3)
    ax.plot(A_e_plot, hi_bound, color=COLORS['upper_bound'], lw=2.3)

    # Bifurcation curves
    if stable:
        ax.scatter([d['A_e'] for d in stable], [d['ln_x'] for d in stable],
                   s=6, c=COLORS['stable'], alpha=0.8, label='Stable', zorder=5)
    if unstable:
        ax.scatter([d['A_e'] for d in unstable], [d['ln_x'] for d in unstable],
                   s=6, c=COLORS['unstable'], alpha=0.6, label='Unstable', zorder=4)

    ax.set_xlabel(r'Cycle affinity $A_{\mathbf{e}}/RT = \ln(a/b)$', fontsize=11)
    ax.set_ylabel(r'$\ln x^{\rm ss}$', fontsize=11)
    ax.set_title('Schlögl Model: Saddle-Node Bifurcation', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 8)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.out}")

    if args.svg:
        svg_out = args.out.replace('.png', '.svg')
        plt.savefig(svg_out, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_out}")

    plt.close()

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
• Bistability emerges via saddle-node bifurcation at A_e/RT ≈ 5.7
• Both stable branches stay WITHIN thermodynamic bounds
• Lower stable branch approaches x = b (lower bound)
• Upper stable branch approaches x = a (upper bound)
• Width of bistable region bounded by exp(A_e)
""")


if __name__ == '__main__':
    main()
