#!/usr/bin/env python3
"""Comprehensive validation of affinity and concentration bounds.

This script validates both affinity bounds and concentration bounds using
LDB-consistent simulations for the self-assembly network from arXiv:2407.11498.

Network:
  R1: F + 2A ⇌ B + W   (fuel-driven dimerization)
  R2: A + B ⇌ C        (trimerization)
  R3: C ⇌ 3A           (disassembly)

Key points:
- Uses fixed chemostat concentrations y = [1, 1] to match bounds assumptions
- Concentration bounds plotted as 2D scatter (concentration vs concentration)
- Affinity bounds plotted per reaction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from crn_bounds.api import CRNInput, run_pipeline
from crn_bounds.self_assembly import self_assembly_crn
from crn_bounds.ldb import sample_rates_from_ldb
from crn_bounds.model import relaxation


# ============================================================================
# Color Palette (clean, consistent styling)
# ============================================================================
COLORS = {
    'bound_fill': '#e8f5e9',      # Very soft mint green
    'upper_bound': '#c0392b',     # Soft brick red
    'lower_bound': '#2980b9',     # Soft steel blue
    'equilibrium': '#27ae60',     # Green for equilibrium line
    'scatter': '#34495e',         # Dark slate gray
    'scatter_alpha': 0.5,
}

# Reaction names in reversible form
RXN_NAMES = [
    r'R1: F + 2A $\rightleftharpoons$ B + W',
    r'R2: A + B $\rightleftharpoons$ C',
    r'R3: C $\rightleftharpoons$ 3A',
]


def compute_reaction_affinity(x, Sx, mu0_X, A_Y):
    """Compute steady-state affinity for each reaction."""
    nR = Sx.shape[1]
    affinities = np.zeros(nR)
    ln_x = np.log(np.maximum(x, 1e-30))
    for rho in range(nR):
        s = Sx[:, rho]
        A_X = -(s @ mu0_X) - (s @ ln_x)
        affinities[rho] = A_X + A_Y[rho]
    return affinities


def sample_steady_states(dm, n_samples=500, seed=42, t_final=500.0, M_tot=15.0):
    """Sample steady states with fixed y = [1, 1]."""
    rng = np.random.default_rng(seed)
    crn = self_assembly_crn()
    mu0_X = np.array([0.0, 0.0, 0.0])
    mu_Y = np.array([dm, 0.0])
    y = np.array([1.0, 1.0])  # Fixed!

    xs = []
    trials = 0
    while len(xs) < n_samples and trials < n_samples * 10:
        trials += 1
        k_plus, k_minus = sample_rates_from_ldb(crn, mu0_X, mu_Y, rng=rng, loga_range=(-3.0, 3.0))
        x2 = np.exp(rng.uniform(-3.0, 2.0))
        x3 = np.exp(rng.uniform(-3.0, 2.0))
        x1 = M_tot - 2 * x2 - 3 * x3
        if x1 <= 1e-8:
            continue
        x0 = np.array([x1, x2, x3])
        try:
            x = relaxation(crn, k_plus, k_minus, y=y, x0=x0, t_final=t_final)
            if np.all(x > 1e-10):
                xs.append(x)
        except:
            continue
    return np.array(xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='notes')
    parser.add_argument('--DeltaMu_over_RT', type=float, default=2.0)
    parser.add_argument('--n', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dm = args.DeltaMu_over_RT

    # Network setup
    Sx = np.array([[-2, -1, 3], [1, -1, 0], [0, 1, -1]], dtype=float)
    mu0_X = np.array([0.0, 0.0, 0.0])
    A_Y = np.array([dm, 0.0, 0.0])

    # Get bounds
    res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

    print("=" * 60)
    print("Self-Assembly Network Bounds Validation")
    print("=" * 60)
    print(f"\nDriving: Δμ/RT = {dm}")
    print("\nReactions:")
    for name in RXN_NAMES:
        print(f"  {name}")

    # Sample
    print(f"\nSampling {args.n} steady states...")
    X = sample_steady_states(dm, n_samples=args.n, seed=args.seed)
    print(f"Got {len(X)} samples")

    # Compute affinities
    affinities = np.array([compute_reaction_affinity(x, Sx, mu0_X, A_Y) for x in X])

    # =========================================================================
    # Figure 1: Affinity Bounds (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for rho in range(3):
        ax = axes[rho]
        lo = res.affinity_bounds[rho].lower
        hi = res.affinity_bounds[rho].upper

        ax.axhspan(lo, hi, alpha=0.35, color=COLORS['bound_fill'], zorder=1)
        ax.axhline(hi, color=COLORS['upper_bound'], lw=2.5,
                   label=f'Upper = {hi:.1f}', zorder=3)
        ax.axhline(lo, color=COLORS['lower_bound'], lw=2.5,
                   label=f'Lower = {lo:.1f}', zorder=3)
        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.4, zorder=2)

        ax.scatter(np.arange(len(affinities)), affinities[:, rho],
                   alpha=COLORS['scatter_alpha'], s=12, color=COLORS['scatter'],
                   edgecolors='none', zorder=4)

        ax.set_xlabel('Sample', fontsize=11)
        ax.set_ylabel(f'$A_{{{rho+1}}}^{{ss}}$ / RT', fontsize=12)
        ax.set_title(RXN_NAMES[rho], fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(lo - 0.3, hi + 0.3)
        ax.grid(True, alpha=0.2, zorder=0)

    plt.suptitle(f'Affinity Bounds Validation (Δμ/RT = {dm})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "fig_affinity_validation.png", dpi=150, bbox_inches='tight')
    plt.savefig(outdir / "fig_affinity_validation.svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {outdir}/fig_affinity_validation.svg")

    # =========================================================================
    # Figure 2: Concentration Bounds (2D scatter plots)
    # Paper style: ln(product) vs ln(reactant)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # R1: 2A ⇌ B, bound on ln(B) vs ln(A²) = 2ln(A)
    # s = [-2, +1, 0] means ln(B/A²) = ln(B) - 2ln(A) ∈ [lo, hi]
    ax = axes[0]
    lb = res.reaction_log_bounds[0]
    lo, hi = lb.lo, lb.hi

    ln_A = np.log(X[:, 0])
    ln_B = np.log(X[:, 1])

    # Plot bounds: ln(B) = 2*ln(A) + offset, offset ∈ [lo, hi]
    x_range = np.linspace(ln_A.min() - 0.5, ln_A.max() + 0.5, 100)
    ax.fill_between(x_range, 2*x_range + lo, 2*x_range + hi,
                    alpha=0.35, color=COLORS['bound_fill'], label='Thermodynamic space')
    ax.plot(x_range, 2*x_range + hi, color=COLORS['upper_bound'], lw=2.5, label='Upper bound')
    ax.plot(x_range, 2*x_range + lo, color=COLORS['lower_bound'], lw=2.5, label='Lower bound')
    ax.plot(x_range, 2*x_range, color=COLORS['equilibrium'], lw=1.5, ls='--',
            alpha=0.7, label='Equilibrium')
    ax.scatter(ln_A, ln_B, alpha=COLORS['scatter_alpha'], s=15,
               color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [A]$', fontsize=12)
    ax.set_ylabel(r'$\ln [B]$', fontsize=12)
    ax.set_title(RXN_NAMES[0], fontsize=11)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # R2: A + B ⇌ C, bound on ln(C) vs ln(AB)
    # s = [-1, -1, +1] means ln(C/AB) = ln(C) - ln(A) - ln(B) ∈ [lo, hi]
    ax = axes[1]
    lb = res.reaction_log_bounds[1]
    lo, hi = lb.lo, lb.hi

    ln_AB = np.log(X[:, 0]) + np.log(X[:, 1])
    ln_C = np.log(X[:, 2])

    x_range = np.linspace(ln_AB.min() - 0.5, ln_AB.max() + 0.5, 100)
    ax.fill_between(x_range, x_range + lo, x_range + hi,
                    alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, x_range + hi, color=COLORS['upper_bound'], lw=2.5)
    ax.plot(x_range, x_range + lo, color=COLORS['lower_bound'], lw=2.5)
    ax.plot(x_range, x_range, color=COLORS['equilibrium'], lw=1.5, ls='--', alpha=0.7)
    ax.scatter(ln_AB, ln_C, alpha=COLORS['scatter_alpha'], s=15,
               color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [A][B]$', fontsize=12)
    ax.set_ylabel(r'$\ln [C]$', fontsize=12)
    ax.set_title(RXN_NAMES[1], fontsize=11)
    ax.grid(True, alpha=0.2)

    # R3: C ⇌ 3A, bound on ln(A³) vs ln(C)
    # s = [-3, 0, +1] (canonical) means ln(C/A³) ∈ [lo, hi]
    # So ln(C) - 3ln(A) ∈ [lo, hi], or ln(A³) = ln(C) - offset
    ax = axes[2]
    lb = res.reaction_log_bounds[2]
    lo, hi = lb.lo, lb.hi

    ln_C = np.log(X[:, 2])
    ln_A3 = 3 * np.log(X[:, 0])

    # Plot as ln(A³) vs ln(C): ln(A³) = ln(C) - offset where offset ∈ [lo, hi]
    # So ln(A³) ∈ [ln(C) - hi, ln(C) - lo]
    x_range = np.linspace(ln_C.min() - 0.5, ln_C.max() + 0.5, 100)
    ax.fill_between(x_range, x_range - hi, x_range - lo,
                    alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, x_range - lo, color=COLORS['upper_bound'], lw=2.5)
    ax.plot(x_range, x_range - hi, color=COLORS['lower_bound'], lw=2.5)
    ax.plot(x_range, x_range, color=COLORS['equilibrium'], lw=1.5, ls='--', alpha=0.7)
    ax.scatter(ln_C, ln_A3, alpha=COLORS['scatter_alpha'], s=15,
               color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [C]$', fontsize=12)
    ax.set_ylabel(r'$\ln [A]^3$', fontsize=12)
    ax.set_title(RXN_NAMES[2], fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.suptitle(f'Concentration Bounds Validation (Δμ/RT = {dm})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "fig_concentration_validation.png", dpi=150, bbox_inches='tight')
    plt.savefig(outdir / "fig_concentration_validation.svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {outdir}/fig_concentration_validation.svg")

    # =========================================================================
    # Figure 3: Combined 2x3 overview
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Affinity bounds
    for rho in range(3):
        ax = axes[0, rho]
        lo = res.affinity_bounds[rho].lower
        hi = res.affinity_bounds[rho].upper
        ax.axhspan(lo, hi, alpha=0.35, color=COLORS['bound_fill'])
        ax.axhline(hi, color=COLORS['upper_bound'], lw=2)
        ax.axhline(lo, color=COLORS['lower_bound'], lw=2)
        ax.scatter(np.arange(len(affinities)), affinities[:, rho],
                   alpha=0.4, s=8, color=COLORS['scatter'], edgecolors='none')
        ax.set_ylabel(f'$A_{{{rho+1}}}^{{ss}}$/RT', fontsize=10)
        ax.set_title(RXN_NAMES[rho], fontsize=10)
        ax.set_ylim(lo - 0.25, hi + 0.25)
        ax.grid(True, alpha=0.2)

    # Row 2: Concentration bounds (2D)
    # R1
    ax = axes[1, 0]
    lb = res.reaction_log_bounds[0]
    ln_A, ln_B = np.log(X[:, 0]), np.log(X[:, 1])
    x_range = np.linspace(ln_A.min() - 0.3, ln_A.max() + 0.3, 50)
    ax.fill_between(x_range, 2*x_range + lb.lo, 2*x_range + lb.hi, alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, 2*x_range + lb.hi, color=COLORS['upper_bound'], lw=2)
    ax.plot(x_range, 2*x_range + lb.lo, color=COLORS['lower_bound'], lw=2)
    ax.scatter(ln_A, ln_B, alpha=0.4, s=8, color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [A]$', fontsize=10)
    ax.set_ylabel(r'$\ln [B]$', fontsize=10)
    ax.grid(True, alpha=0.2)

    # R2
    ax = axes[1, 1]
    lb = res.reaction_log_bounds[1]
    ln_AB, ln_C = np.log(X[:, 0]) + np.log(X[:, 1]), np.log(X[:, 2])
    x_range = np.linspace(ln_AB.min() - 0.3, ln_AB.max() + 0.3, 50)
    ax.fill_between(x_range, x_range + lb.lo, x_range + lb.hi, alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, x_range + lb.hi, color=COLORS['upper_bound'], lw=2)
    ax.plot(x_range, x_range + lb.lo, color=COLORS['lower_bound'], lw=2)
    ax.scatter(ln_AB, ln_C, alpha=0.4, s=8, color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [A][B]$', fontsize=10)
    ax.set_ylabel(r'$\ln [C]$', fontsize=10)
    ax.grid(True, alpha=0.2)

    # R3
    ax = axes[1, 2]
    lb = res.reaction_log_bounds[2]
    ln_C, ln_A3 = np.log(X[:, 2]), 3 * np.log(X[:, 0])
    x_range = np.linspace(ln_C.min() - 0.3, ln_C.max() + 0.3, 50)
    ax.fill_between(x_range, x_range - lb.hi, x_range - lb.lo, alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, x_range - lb.lo, color=COLORS['upper_bound'], lw=2)
    ax.plot(x_range, x_range - lb.hi, color=COLORS['lower_bound'], lw=2)
    ax.scatter(ln_C, ln_A3, alpha=0.4, s=8, color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [C]$', fontsize=10)
    ax.set_ylabel(r'$\ln [A]^3$', fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.text(0.02, 0.72, 'Affinity', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.28, 'Concentration', fontsize=12, fontweight='bold', rotation=90, va='center')

    plt.suptitle(f'Self-Assembly Network: Bounds Validation (Δμ/RT = {dm})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig(outdir / "fig_bounds_combined.png", dpi=150, bbox_inches='tight')
    plt.savefig(outdir / "fig_bounds_combined.svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {outdir}/fig_bounds_combined.svg")

    # Validation check
    print("\nValidation:")
    for rho in range(3):
        lo_A, hi_A = res.affinity_bounds[rho].lower, res.affinity_bounds[rho].upper
        viol = np.sum((affinities[:, rho] < lo_A - 1e-6) | (affinities[:, rho] > hi_A + 1e-6))
        print(f"  R{rho+1} Affinity: {'✓' if viol == 0 else '✗'}")


if __name__ == '__main__':
    main()
