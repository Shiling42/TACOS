#!/usr/bin/env python3
"""Visualize affinity bounds with simulation scatter validation.

For each reaction, plot:
- The theoretical affinity bounds from EFM analysis
- Scatter points from LDB-consistent simulations showing actual steady-state affinities

Reference: arXiv:2407.11498 (Thermodynamic space for CRNs)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from crn_bounds.api import CRNInput, run_pipeline
from crn_bounds.self_assembly import self_assembly_crn
from crn_bounds.ldb import sample_rates_from_ldb
from crn_bounds.model import relaxation


# ============================================================================
# Color Palette (clean, low-saturation colors)
# ============================================================================
COLORS = {
    'bound_fill': '#d4edda',      # Soft mint green (low saturation)
    'bound_fill_alpha': 0.4,
    'upper_bound': '#c0392b',     # Soft brick red
    'lower_bound': '#2980b9',     # Soft steel blue
    'scatter': '#34495e',         # Dark slate gray
    'scatter_alpha': 0.45,
    'grid': '#bdc3c7',            # Light gray
}


def compute_reaction_affinity(
    x: np.ndarray,
    Sx: np.ndarray,
    mu0_X: np.ndarray,
    A_Y: np.ndarray,
) -> np.ndarray:
    """Compute steady-state affinity for each reaction.

    A_rho = A_rho^X + A_rho^Y
    A_rho^X = -ΔG_rho^0 - s^T ln x = -(s^T μ0) - s^T ln x
    """
    nR = Sx.shape[1]
    affinities = np.zeros(nR)
    ln_x = np.log(np.maximum(x, 1e-30))

    for rho in range(nR):
        s = Sx[:, rho]
        dG0 = s @ mu0_X
        A_X = -dG0 - s @ ln_x
        A_rho = A_X + A_Y[rho]
        affinities[rho] = A_rho

    return affinities


def sample_steady_states(
    dm: float,
    n_samples: int = 500,
    seed: int = 42,
    t_final: float = 300.0,
    M_tot: float = 15.0,
    loga_range: tuple = (-3.0, 3.0),  # Wider range for better coverage
) -> np.ndarray:
    """Sample steady states from LDB-consistent rate sampling.

    Uses wider parameter ranges to achieve saturated bound coverage.
    """
    rng = np.random.default_rng(seed)

    crn = self_assembly_crn()
    mu0_X = np.array([0.0, 0.0, 0.0])
    mu_Y = np.array([dm, 0.0])  # [F, W] -> Δμ = μ_F - μ_W = dm

    xs = []
    kept = 0
    trials = 0
    max_trials = n_samples * 8

    while kept < n_samples and trials < max_trials:
        trials += 1

        # Wider range of rate constants for better bound saturation
        k_plus, k_minus = sample_rates_from_ldb(
            crn, mu0_X, mu_Y, rng=rng, loga_range=loga_range
        )

        # Vary chemostat concentrations for more diversity
        y = np.array([
            np.exp(rng.uniform(-1.0, 1.5)),  # F concentration
            np.exp(rng.uniform(-1.0, 1.5)),  # W concentration
        ])

        # Random initial condition with wider range
        x2 = np.exp(rng.uniform(-3.0, 2.0))
        x3 = np.exp(rng.uniform(-3.0, 2.0))
        x1 = M_tot - 2 * x2 - 3 * x3
        if x1 <= 1e-8:
            continue
        x0 = np.array([x1, x2, x3], dtype=float)

        try:
            x = relaxation(crn, k_plus, k_minus, y=y, x0=x0, t_final=t_final)
            if np.all(x > 0):
                xs.append(x)
                kept += 1
        except Exception:
            continue

    return np.array(xs, dtype=float) if xs else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Plot affinity bounds with simulation validation")
    parser.add_argument('--outdir', type=str, default='notes',
                       help='Output directory')
    parser.add_argument('--DeltaMu_over_RT', type=float, default=2.0,
                       help='Driving magnitude Δμ/RT')
    parser.add_argument('--n', type=int, default=500,
                       help='Number of simulation samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dm = args.DeltaMu_over_RT

    # Self-assembly stoichiometry (net)
    Sx = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=float)

    mu0_X = np.array([0.0, 0.0, 0.0])
    A_Y = np.array([dm, 0.0, 0.0])  # External affinity contribution

    # Get theoretical bounds
    res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

    # Sample steady states with wider parameter range
    print(f"Sampling {args.n} steady states with wide parameter range...")
    X = sample_steady_states(dm, n_samples=args.n, seed=args.seed)
    print(f"Got {len(X)} valid samples")

    if len(X) == 0:
        print("No valid samples, exiting.")
        return

    # Compute affinities at each steady state
    affinities = np.array([compute_reaction_affinity(x, Sx, mu0_X, A_Y) for x in X])

    # ===== Figure 1: Individual reaction panels =====
    rxn_names = ['R1: 2A → B', 'R2: A+B → C', 'R3: C → 3A']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for rho in range(3):
        ax = axes[rho]

        lo = res.affinity_bounds[rho].lower
        hi = res.affinity_bounds[rho].upper

        # Plot bound region with soft green fill
        ax.axhspan(lo, hi, alpha=COLORS['bound_fill_alpha'],
                   color=COLORS['bound_fill'], label='Thermodynamic space')

        # Upper and lower bounds with different colors
        ax.axhline(hi, color=COLORS['upper_bound'], linewidth=2.5,
                   linestyle='-', label='Upper bound')
        ax.axhline(lo, color=COLORS['lower_bound'], linewidth=2.5,
                   linestyle='-', label='Lower bound')
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.4)

        # Plot scatter points
        y_vals = affinities[:, rho]
        x_vals = np.arange(len(y_vals))
        ax.scatter(x_vals, y_vals, alpha=COLORS['scatter_alpha'], s=18,
                   color=COLORS['scatter'], label='Simulated', edgecolors='none')

        ax.set_xlabel('Sample index', fontsize=11)
        ax.set_ylabel(f'$A_{{{rho+1}}}^{{ss}}$ / RT', fontsize=11)
        ax.set_title(f'{rxn_names[rho]}', fontsize=12, fontweight='medium')

        if rho == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        ax.set_ylim(lo - 0.4, hi + 0.4)
        ax.grid(True, alpha=0.2, color=COLORS['grid'])

    plt.suptitle(f'Self-assembly affinity bounds (Δμ/RT = {dm})', fontsize=14, y=1.02)
    plt.tight_layout()

    outfile = outdir / "fig_affinity_bounds_self_assembly.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    svg_file = outdir / "fig_affinity_bounds_self_assembly.svg"
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {outfile}")
    print(f"Wrote: {svg_file}")

    # ===== Figure 2: Bar chart with scatter overlay =====
    fig, ax = plt.subplots(figsize=(8, 5.5))

    rxn_labels = ['R1\n(2A→B)', 'R2\n(A+B→C)', 'R3\n(C→3A)']
    x_pos = np.array([0, 1, 2])
    bar_width = 0.65

    # Plot bound bars
    for rho in range(3):
        lo = res.affinity_bounds[rho].lower
        hi = res.affinity_bounds[rho].upper
        height = hi - lo

        # Soft green fill
        rect = Rectangle((x_pos[rho] - bar_width/2, lo), bar_width, height,
                         facecolor=COLORS['bound_fill'],
                         edgecolor='none',
                         alpha=0.6,
                         label='Thermodynamic space' if rho == 0 else '')
        ax.add_patch(rect)

        # Upper bound line (red)
        ax.hlines(hi, x_pos[rho] - bar_width/2, x_pos[rho] + bar_width/2,
                  color=COLORS['upper_bound'], linewidth=3,
                  label='Upper bound' if rho == 0 else '')

        # Lower bound line (blue)
        ax.hlines(lo, x_pos[rho] - bar_width/2, x_pos[rho] + bar_width/2,
                  color=COLORS['lower_bound'], linewidth=3,
                  label='Lower bound' if rho == 0 else '')

    # Plot scatter points with jitter
    rng_jitter = np.random.default_rng(args.seed)
    for rho in range(3):
        jitter = rng_jitter.uniform(-0.2, 0.2, len(affinities))
        ax.scatter(x_pos[rho] + jitter, affinities[:, rho],
                  alpha=COLORS['scatter_alpha'], s=12,
                  color=COLORS['scatter'],
                  edgecolors='none',
                  label='Simulated' if rho == 0 else '')

    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(rxn_labels, fontsize=11)
    ax.set_ylabel('Reaction Affinity $A_\\rho^{ss}$ / RT', fontsize=12)
    ax.set_title(f'Self-assembly: Affinity bounds vs simulations (Δμ/RT = {dm})', fontsize=13)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.4, dm + 0.4)
    ax.grid(True, axis='y', alpha=0.2, color=COLORS['grid'])

    plt.tight_layout()

    outfile = outdir / "fig_affinity_bounds_bar.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    svg_file = outdir / "fig_affinity_bounds_bar.svg"
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {outfile}")
    print(f"Wrote: {svg_file}")

    # ===== Figure 3: 2D scatter of affinity pairs =====
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    pairs = [(0, 1), (1, 2), (0, 2)]
    pair_labels = [('$A_1^{ss}$', '$A_2^{ss}$'),
                   ('$A_2^{ss}$', '$A_3^{ss}$'),
                   ('$A_1^{ss}$', '$A_3^{ss}$')]

    for idx, ((i, j), (xlabel, ylabel)) in enumerate(zip(pairs, pair_labels)):
        ax = axes[idx]

        # Draw bound rectangle
        lo_i, hi_i = res.affinity_bounds[i].lower, res.affinity_bounds[i].upper
        lo_j, hi_j = res.affinity_bounds[j].lower, res.affinity_bounds[j].upper

        # Soft green fill
        rect = Rectangle((lo_i, lo_j), hi_i - lo_i, hi_j - lo_j,
                         facecolor=COLORS['bound_fill'],
                         edgecolor='none',
                         alpha=0.5,
                         label='Thermodynamic space')
        ax.add_patch(rect)

        # Bound edges with different colors
        # Horizontal lines (for j axis)
        ax.hlines(hi_j, lo_i, hi_i, color=COLORS['upper_bound'], linewidth=2.5)
        ax.hlines(lo_j, lo_i, hi_i, color=COLORS['lower_bound'], linewidth=2.5)
        # Vertical lines (for i axis)
        ax.vlines(hi_i, lo_j, hi_j, color=COLORS['upper_bound'], linewidth=2.5)
        ax.vlines(lo_i, lo_j, hi_j, color=COLORS['lower_bound'], linewidth=2.5)

        # Plot scatter
        ax.scatter(affinities[:, i], affinities[:, j],
                  alpha=COLORS['scatter_alpha'], s=18,
                  color=COLORS['scatter'],
                  edgecolors='none',
                  label='Simulated')

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)

        ax.set_xlabel(xlabel + ' / RT', fontsize=11)
        ax.set_ylabel(ylabel + ' / RT', fontsize=11)
        ax.set_xlim(lo_i - 0.25, hi_i + 0.25)
        ax.set_ylim(lo_j - 0.25, hi_j + 0.25)
        ax.grid(True, alpha=0.15, color=COLORS['grid'])

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.suptitle(f'Self-assembly: Pairwise affinity bounds (Δμ/RT = {dm})', fontsize=13, y=1.02)
    plt.tight_layout()

    outfile = outdir / "fig_affinity_bounds_2d.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    svg_file = outdir / "fig_affinity_bounds_2d.svg"
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {outfile}")
    print(f"Wrote: {svg_file}")


if __name__ == '__main__':
    main()
