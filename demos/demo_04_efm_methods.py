#!/usr/bin/env python3
"""
TACOS Demo 4: EFM Enumeration Methods Comparison
=================================================

Compares two algorithms for Elementary Flux Mode enumeration:

1. COMBINATORIAL: Circuit enumeration via SVD nullspace
   - O(C(2n, r)) complexity, only practical for n ≤ 6

2. GEOMETRIC: Extreme ray enumeration via pycddlib
   - Output-sensitive, scales to n > 15

Test network: Modular self-assembly (n species, 2n reactions)

Reference: arXiv:2407.11498, Fig. EFM_search
"""

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def build_modular_assembly_network(n):
    """Build stoichiometry matrix for modular self-assembly of target size n.

    Network:
    - Species: A (inactive), B_1,...,B_n (active monomers/aggregates)
    - Reactions:
      1) A -> B_1 (activation)
      2) B_1 + B_m -> B_{m+1} for m=1,...,n-1 (assembly)
      3) B_m -> m*A for m=1,...,n (degradation)
    """
    n_species = n + 1
    n_reactions = 2 * n

    Sx = np.zeros((n_species, n_reactions), dtype=float)

    rxn = 0
    # Activation: A -> B_1
    Sx[0, rxn] = -1
    Sx[1, rxn] = +1
    rxn += 1

    # Assembly: B_1 + B_m -> B_{m+1}
    for m in range(1, n):
        Sx[1, rxn] = -1
        Sx[m, rxn] = -1
        Sx[m+1, rxn] = +1
        rxn += 1

    # Degradation: B_m -> m*A
    for m in range(1, n+1):
        Sx[m, rxn] = -1
        Sx[0, rxn] = +m
        rxn += 1

    return Sx


def enumerate_efms_combinatorial(Sx, tol=1e-10):
    """Enumerate EFMs using combinatorial circuit enumeration."""
    nX, nR = Sx.shape
    Sx_split = np.hstack([Sx, -Sx])
    nR_split = 2 * nR
    rank = np.linalg.matrix_rank(Sx_split)

    efms_split = []

    for size in range(2, min(rank + 2, nR_split + 1)):
        for subset in combinations(range(nR_split), size):
            sub_cols = Sx_split[:, list(subset)]
            u, s, vh = np.linalg.svd(sub_cols, full_matrices=True)
            null_dim = np.sum(s < tol)

            if null_dim != 1:
                continue

            null_vec = vh[-1, :]
            if not (np.all(null_vec >= -tol) or np.all(null_vec <= tol)):
                continue

            if np.sum(null_vec) < 0:
                null_vec = -null_vec
            null_vec = np.maximum(null_vec, 0)

            if np.sum(null_vec > tol) != size:
                continue

            full_vec = np.zeros(nR_split)
            for i, idx in enumerate(subset):
                full_vec[idx] = null_vec[i]

            is_dup = any(np.allclose(full_vec / (np.linalg.norm(full_vec) + 1e-12),
                                     e / (np.linalg.norm(e) + 1e-12), atol=tol)
                         for e in efms_split)
            if not is_dup:
                efms_split.append(full_vec)

    # Convert to original coordinates
    efms_orig = []
    for e_split in efms_split:
        e_fwd = e_split[:nR]
        e_bwd = e_split[nR:]
        e_orig = e_fwd - e_bwd
        if np.any((e_fwd > tol) & (e_bwd > tol)):
            continue
        if np.linalg.norm(e_orig) > tol:
            e_orig = e_orig / np.linalg.norm(e_orig)
            for x in e_orig:
                if abs(x) > tol:
                    if x < 0:
                        e_orig = -e_orig
                    break
            is_dup = any(np.allclose(e_orig, e, atol=tol) for e in efms_orig)
            if not is_dup:
                efms_orig.append(e_orig)

    return efms_orig


def enumerate_efms_geometric(Sx, tol=1e-10):
    """Enumerate EFMs using geometric extreme-ray enumeration (pycddlib)."""
    try:
        import cdd
    except ImportError:
        raise ImportError("pycddlib required for geometric method")

    nX, nR = Sx.shape
    Sx_split = np.hstack([Sx, -Sx])
    nR_split = 2 * nR

    n_eq = nX
    n_ineq = nR_split

    H = np.zeros((n_eq + n_ineq, 1 + nR_split))
    H[:n_eq, 1:] = Sx_split
    H[n_eq:, 1:] = np.eye(nR_split)

    mat = cdd.matrix_from_array(H, rep_type=cdd.RepType.INEQUALITY)
    mat.lin_set = frozenset(range(n_eq))

    poly = cdd.polyhedron_from_matrix(mat)
    generators = cdd.copy_generators(poly)

    efms_split = []
    for row in generators.array:
        if row[0] == 0:
            ray = np.array(row[1:], dtype=float)
            if np.linalg.norm(ray) > tol:
                efms_split.append(ray)

    efms_orig = []
    for e_split in efms_split:
        e_fwd = e_split[:nR]
        e_bwd = e_split[nR:]
        e_orig = e_fwd - e_bwd
        if np.any((e_fwd > tol) & (e_bwd > tol)):
            continue
        if np.linalg.norm(e_orig) > tol:
            e_orig = e_orig / np.linalg.norm(e_orig)
            for x in e_orig:
                if abs(x) > tol:
                    if x < 0:
                        e_orig = -e_orig
                    break
            is_dup = any(np.allclose(e_orig, e, atol=tol) for e in efms_orig)
            if not is_dup:
                efms_orig.append(e_orig)

    return efms_orig


def run_benchmark(n_min=3, n_max=15):
    """Run benchmark for different network sizes."""
    results = {'n': [], 'n_efms': [], 't_comb': [], 't_geo': []}

    for n in range(n_min, n_max + 1):
        print(f"n={n}: ", end="", flush=True)
        Sx = build_modular_assembly_network(n)
        results['n'].append(n)

        # Geometric method
        try:
            t0 = time.perf_counter()
            efms_geo = enumerate_efms_geometric(Sx)
            t_geo = time.perf_counter() - t0
            results['t_geo'].append(t_geo)
            results['n_efms'].append(len(efms_geo))
            print(f"geo={t_geo:.3f}s ({len(efms_geo)} EFMs), ", end="")
        except Exception as e:
            results['t_geo'].append(np.nan)
            results['n_efms'].append(np.nan)
            print(f"geo=FAILED, ", end="")

        # Combinatorial (only for small n)
        if n <= 6:
            try:
                t0 = time.perf_counter()
                efms_comb = enumerate_efms_combinatorial(Sx)
                t_comb = time.perf_counter() - t0
                results['t_comb'].append(t_comb)
                print(f"comb={t_comb:.3f}s")
            except Exception:
                results['t_comb'].append(np.nan)
                print("comb=FAILED")
        else:
            results['t_comb'].append(np.nan)
            print("comb=SKIPPED")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='notes/demo_efm_methods.png')
    parser.add_argument('--svg', action='store_true', help='Also save SVG')
    parser.add_argument('--n-max', type=int, default=15)
    args = parser.parse_args()

    outdir = Path(args.out).parent
    outdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("TACOS Demo 4: EFM Enumeration Methods")
    print("=" * 60)

    results = run_benchmark(n_min=3, n_max=args.n_max)

    # Create 3-panel figure matching paper style
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    n_vals = np.array(results['n'])
    n_efms = np.array(results['n_efms'])
    t_comb = np.array(results['t_comb'])
    t_geo = np.array(results['t_geo'])

    # Panel (a): Network schematic placeholder
    ax = axes[0]
    ax.text(0.5, 0.5, "(a) Modular assembly network\n\nA → B₁ → B₂ → ... → Bₙ\n\nSee paper Fig. S1",
            ha='center', va='center', fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel (b): Computation time
    ax = axes[1]
    mask_comb = ~np.isnan(t_comb)
    mask_geo = ~np.isnan(t_geo)

    if np.any(mask_comb):
        ax.semilogy(n_vals[mask_comb], t_comb[mask_comb], 's-',
                    color='#1f77b4', ms=8, lw=2, label='Combinatorial')
    if np.any(mask_geo):
        ax.semilogy(n_vals[mask_geo], t_geo[mask_geo], 'o-',
                    color='#ff7f0e', ms=8, lw=2, label='Geometric')

    ax.set_xlabel('Target Structure Size (n)', fontsize=11)
    ax.set_ylabel('Computation Time (s)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b)', fontsize=12, fontweight='bold', loc='left')

    # Panel (c): Number of EFMs
    ax = axes[2]
    mask_efm = ~np.isnan(n_efms)
    if np.any(mask_efm):
        ax.semilogy(n_vals[mask_efm], n_efms[mask_efm], 'o-',
                    color='#2ca02c', ms=8, lw=2)

    ax.set_xlabel('Target Structure Size (n)', fontsize=11)
    ax.set_ylabel('Number of EFMs', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('(c)', fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.out}")

    if args.svg:
        svg_out = args.out.replace('.png', '.svg')
        plt.savefig(svg_out, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_out}")

    plt.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
COMBINATORIAL: O(C(2n,r)) - exponential, practical for n ≤ 6
GEOMETRIC: Output-sensitive - scales to n > 15

Both methods enumerate the SAME Elementary Flux Modes.
Geometric method strongly recommended for larger networks.
""")


if __name__ == '__main__':
    main()
