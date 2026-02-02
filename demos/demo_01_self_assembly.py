#!/usr/bin/env python3
"""
TACOS Demo 1: Self-Assembly Network
====================================

Fuel-driven self-assembly network from arXiv:2407.11498.

Network:
  R1: F + 2A ⇌ B + W   (fuel-driven dimerization)
  R2: A + B ⇌ C        (trimerization)
  R3: C ⇌ 3A           (disassembly)

This demo computes thermodynamic bounds AND validates them with
LDB-consistent simulations showing scatter points within the bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from crn_bounds.api import CRNInput, run_pipeline
from crn_bounds.self_assembly import self_assembly_crn
from crn_bounds.ldb import sample_rates_from_ldb
from crn_bounds.model import relaxation

# ============================================================================
# Consistent Color Palette
# ============================================================================
COLORS = {
    'bound_fill': '#e8f5e9',
    'upper_bound': '#c0392b',
    'lower_bound': '#2980b9',
    'equilibrium': '#27ae60',
    'scatter': '#34495e',
}

RXN_NAMES = [
    r'R1: F + 2A $\rightleftharpoons$ B + W',
    r'R2: A + B $\rightleftharpoons$ C',
    r'R3: C $\rightleftharpoons$ 3A',
]


def sample_steady_states(dm, n_samples=500, seed=42):
    """Sample steady states with LDB-consistent kinetics."""
    rng = np.random.default_rng(seed)
    crn = self_assembly_crn()
    mu0_X = np.array([0.0, 0.0, 0.0])
    mu_Y = np.array([dm, 0.0])
    y = np.array([1.0, 1.0])  # Fixed chemostat

    xs = []
    trials = 0
    while len(xs) < n_samples and trials < n_samples * 10:
        trials += 1
        k_plus, k_minus = sample_rates_from_ldb(crn, mu0_X, mu_Y, rng=rng, loga_range=(-3.0, 3.0))
        x2 = np.exp(rng.uniform(-3.0, 2.0))
        x3 = np.exp(rng.uniform(-3.0, 2.0))
        x1 = 15.0 - 2 * x2 - 3 * x3
        if x1 <= 1e-8:
            continue
        x0 = np.array([x1, x2, x3])
        try:
            x = relaxation(crn, k_plus, k_minus, y=y, x0=x0, t_final=500.0)
            if np.all(x > 1e-10):
                xs.append(x)
        except:
            continue
    return np.array(xs)


def main():
    print("=" * 70)
    print("TACOS Demo 1: Self-Assembly Network")
    print("=" * 70)

    # Network setup
    Sx = np.array([[-2, -1, 3], [1, -1, 0], [0, 1, -1]], dtype=float)
    mu0_X = np.array([0.0, 0.0, 0.0])
    delta_mu = 2.0
    A_Y = np.array([delta_mu, 0.0, 0.0])

    print("\nNetwork:")
    for name in RXN_NAMES:
        print(f"  {name}")
    print(f"\nDriving: Δμ/RT = {delta_mu}")

    # Compute bounds
    print("\nComputing thermodynamic bounds...")
    result = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

    print("\nAffinity Bounds:")
    for i, ab in enumerate(result.affinity_bounds):
        print(f"  R{i+1}: A ∈ [{ab.lower:.3f}, {ab.upper:.3f}]")

    print("\nConcentration Bounds:")
    for i, lb in enumerate(result.reaction_log_bounds):
        print(f"  R{i+1}: s·ln(x) ∈ [{lb.lo:.3f}, {lb.hi:.3f}]")

    # Sample steady states
    print("\nSampling 800 steady states with LDB-consistent kinetics...")
    X = sample_steady_states(delta_mu, n_samples=800, seed=42)
    print(f"  Got {len(X)} valid samples")

    # Compute affinities
    affinities = []
    for x in X:
        A = []
        ln_x = np.log(x)
        for rho in range(3):
            s = Sx[:, rho]
            A_rho = -(s @ mu0_X) - (s @ ln_x) + A_Y[rho]
            A.append(A_rho)
        affinities.append(A)
    affinities = np.array(affinities)

    # =========================================================================
    # Figure: Combined affinity and concentration validation
    # =========================================================================
    outdir = Path("notes")
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Affinity bounds with scatter
    for rho in range(3):
        ax = axes[0, rho]
        lo = result.affinity_bounds[rho].lower
        hi = result.affinity_bounds[rho].upper

        ax.axhspan(lo, hi, alpha=0.35, color=COLORS['bound_fill'])
        ax.axhline(hi, color=COLORS['upper_bound'], lw=2.5, label=f'Upper = {hi:.1f}')
        ax.axhline(lo, color=COLORS['lower_bound'], lw=2.5, label=f'Lower = {lo:.1f}')
        ax.scatter(np.arange(len(affinities)), affinities[:, rho],
                   alpha=0.5, s=10, color=COLORS['scatter'], edgecolors='none')

        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel(f'$A_{{{rho+1}}}^{{ss}}$/RT', fontsize=11)
        ax.set_title(RXN_NAMES[rho], fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(lo - 0.3, hi + 0.3)
        ax.grid(True, alpha=0.2)

    # Row 2: Concentration bounds (2D scatter)
    # R1: ln(B) vs ln(A)
    ax = axes[1, 0]
    lb = result.reaction_log_bounds[0]
    ln_A, ln_B = np.log(X[:, 0]), np.log(X[:, 1])
    x_range = np.linspace(ln_A.min() - 0.3, ln_A.max() + 0.3, 100)
    ax.fill_between(x_range, 2*x_range + lb.lo, 2*x_range + lb.hi, alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, 2*x_range + lb.hi, color=COLORS['upper_bound'], lw=2.5)
    ax.plot(x_range, 2*x_range + lb.lo, color=COLORS['lower_bound'], lw=2.5)
    ax.plot(x_range, 2*x_range, color=COLORS['equilibrium'], lw=1.5, ls='--', alpha=0.7)
    ax.scatter(ln_A, ln_B, alpha=0.5, s=10, color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [A]$', fontsize=11)
    ax.set_ylabel(r'$\ln [B]$', fontsize=11)
    ax.set_title(RXN_NAMES[0], fontsize=11)
    ax.grid(True, alpha=0.2)

    # R2: ln(C) vs ln(AB)
    ax = axes[1, 1]
    lb = result.reaction_log_bounds[1]
    ln_AB, ln_C = np.log(X[:, 0]) + np.log(X[:, 1]), np.log(X[:, 2])
    x_range = np.linspace(ln_AB.min() - 0.3, ln_AB.max() + 0.3, 100)
    ax.fill_between(x_range, x_range + lb.lo, x_range + lb.hi, alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, x_range + lb.hi, color=COLORS['upper_bound'], lw=2.5)
    ax.plot(x_range, x_range + lb.lo, color=COLORS['lower_bound'], lw=2.5)
    ax.plot(x_range, x_range, color=COLORS['equilibrium'], lw=1.5, ls='--', alpha=0.7)
    ax.scatter(ln_AB, ln_C, alpha=0.5, s=10, color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [A][B]$', fontsize=11)
    ax.set_ylabel(r'$\ln [C]$', fontsize=11)
    ax.set_title(RXN_NAMES[1], fontsize=11)
    ax.grid(True, alpha=0.2)

    # R3: ln(A³) vs ln(C)
    ax = axes[1, 2]
    lb = result.reaction_log_bounds[2]
    ln_C, ln_A3 = np.log(X[:, 2]), 3 * np.log(X[:, 0])
    x_range = np.linspace(ln_C.min() - 0.3, ln_C.max() + 0.3, 100)
    ax.fill_between(x_range, x_range - lb.hi, x_range - lb.lo, alpha=0.35, color=COLORS['bound_fill'])
    ax.plot(x_range, x_range - lb.lo, color=COLORS['upper_bound'], lw=2.5)
    ax.plot(x_range, x_range - lb.hi, color=COLORS['lower_bound'], lw=2.5)
    ax.plot(x_range, x_range, color=COLORS['equilibrium'], lw=1.5, ls='--', alpha=0.7)
    ax.scatter(ln_C, ln_A3, alpha=0.5, s=10, color=COLORS['scatter'], edgecolors='none')
    ax.set_xlabel(r'$\ln [C]$', fontsize=11)
    ax.set_ylabel(r'$\ln [A]^3$', fontsize=11)
    ax.set_title(RXN_NAMES[2], fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.text(0.02, 0.72, 'Affinity', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.28, 'Concentration', fontsize=12, fontweight='bold', rotation=90, va='center')

    plt.suptitle(f'Self-Assembly: Thermodynamic Bounds with Simulation Validation (Δμ/RT = {delta_mu})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    plt.savefig(outdir / "demo_self_assembly.png", dpi=150, bbox_inches='tight')
    plt.savefig(outdir / "demo_self_assembly.svg", format='svg', bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {outdir}/demo_self_assembly.svg")

    # Validation check
    print("\nValidation:")
    for rho in range(3):
        lo, hi = result.affinity_bounds[rho].lower, result.affinity_bounds[rho].upper
        viol = np.sum((affinities[:, rho] < lo - 1e-6) | (affinities[:, rho] > hi + 1e-6))
        print(f"  R{rho+1}: {'✓ All within bounds' if viol == 0 else f'✗ {viol} violations'}")


if __name__ == '__main__':
    main()
