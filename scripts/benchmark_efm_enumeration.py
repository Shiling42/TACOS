#!/usr/bin/env python3
"""Benchmark EFM enumeration: combinatorial vs geometric methods.

This script compares two EFM enumeration algorithms:
1. Combinatorial (subset-based circuit enumeration)
2. Geometric (polyhedral extreme-ray via pycddlib)

on a modular self-assembly network of increasing size n.

Reference: arXiv:2407.11498 Appendix A (Fig. S1)
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


def build_modular_assembly_network(n: int) -> NDArray[np.float64]:
    """Build stoichiometry matrix for modular self-assembly of target size n.

    Network:
    - Species: A (inactive), B_1,...,B_n (active monomers/aggregates)
    - Reactions:
      1) F + A <-> B_1 + W  (fuel-driven activation)
      2) B_1 + B_m <-> B_{m+1} for m=1,...,n-1 (assembly)
      3) B_m <-> m*A for m=2,...,n (degradation)

    Returns:
        Sx: (n+1) x (2n) internal stoichiometry matrix
    """
    n_species = n + 1  # A, B_1, ..., B_n
    n_reactions = 2 * n  # 1 activation + (n-1) assembly + n degradation

    Sx = np.zeros((n_species, n_reactions), dtype=float)

    # Species indices: 0=A, 1=B_1, 2=B_2, ..., n=B_n
    # Reaction indices:
    # 0: F + A -> B_1 + W (activation)
    # 1 to n-1: B_1 + B_m -> B_{m+1} (assembly, m=1,...,n-1)
    # n to 2n-1: B_m -> m*A (degradation, m=2,...,n) -- actually we do m=1,...,n

    # Let me restructure:
    # Reactions 0 to n-1: activation + assembly
    # Reactions n to 2n-1: degradation

    # Actually, following the paper more closely:
    # R0: A -> B_1 (activation, driven by F->W)
    # R1 to R_{n-1}: B_1 + B_m -> B_{m+1} for m=1,...,n-1
    # R_n to R_{2n-1}: B_m -> m*A for m=2,...,n (but paper says m=2,...,n which is n-1 reactions)

    # Wait, paper says: "number of reactions in S^X equal to 2n"
    # R0: activation (1 reaction)
    # R1 to R_{n-1}: assembly (n-1 reactions)
    # R_n to R_{2n-1}: degradation (n reactions, for m=1,...,n)
    # Total: 1 + (n-1) + n = 2n ✓

    # But actually paper says degradation is for m=2,...,n, which is n-1 reactions
    # 1 + (n-1) + (n-1) = 2n-1, not 2n
    # Let me re-read: "B_m -> mA, m=2,...,n"
    # That's n-1 reactions. With 1 activation and n-1 assembly = 2n-1.

    # Hmm, let me just follow the network structure and count:
    # - 1 activation: A -> B_1
    # - n-1 assembly: B_1+B_1->B_2, B_1+B_2->B_3, ..., B_1+B_{n-1}->B_n
    # - n degradation: B_1->A, B_2->2A, ..., B_n->nA
    # Total: 1 + (n-1) + n = 2n ✓

    rxn = 0

    # Activation: A -> B_1
    Sx[0, rxn] = -1  # A consumed
    Sx[1, rxn] = +1  # B_1 produced
    rxn += 1

    # Assembly: B_1 + B_m -> B_{m+1} for m=1,...,n-1
    for m in range(1, n):
        Sx[1, rxn] = -1      # B_1 consumed
        Sx[m, rxn] = -1      # B_m consumed
        Sx[m+1, rxn] = +1    # B_{m+1} produced
        rxn += 1

    # Degradation: B_m -> m*A for m=1,...,n
    for m in range(1, n+1):
        Sx[m, rxn] = -1      # B_m consumed
        Sx[0, rxn] = +m      # m copies of A produced
        rxn += 1

    assert rxn == 2*n
    return Sx


def enumerate_efms_combinatorial(Sx: NDArray[np.float64], tol: float = 1e-10) -> list[NDArray[np.float64]]:
    """Enumerate EFMs using combinatorial circuit enumeration.

    For split representation, find minimal subsets of columns whose nullspace
    is 1-dimensional and spanned by a non-negative vector.
    """
    nX, nR = Sx.shape

    # Build split matrix: [Sx, -Sx]
    Sx_split = np.hstack([Sx, -Sx])
    nR_split = 2 * nR

    rank = np.linalg.matrix_rank(Sx_split)

    efms_split = []

    # Iterate over subsets of size 2 to rank+1
    for size in range(2, min(rank + 2, nR_split + 1)):
        for subset in combinations(range(nR_split), size):
            sub_cols = Sx_split[:, list(subset)]

            # Check if nullspace is 1-dimensional
            u, s, vh = np.linalg.svd(sub_cols, full_matrices=True)
            null_dim = np.sum(s < tol)

            if null_dim != 1:
                continue

            # Get null vector
            null_vec = vh[-1, :]

            # Check if all entries have same sign (can be made non-negative)
            if not (np.all(null_vec >= -tol) or np.all(null_vec <= tol)):
                continue

            # Make non-negative
            if np.sum(null_vec) < 0:
                null_vec = -null_vec
            null_vec = np.maximum(null_vec, 0)

            # Check full support
            if np.sum(null_vec > tol) != size:
                continue

            # Embed into full split space
            full_vec = np.zeros(nR_split)
            for i, idx in enumerate(subset):
                full_vec[idx] = null_vec[i]

            # Check not already found (up to scaling)
            is_dup = False
            for existing in efms_split:
                if np.allclose(full_vec / (np.linalg.norm(full_vec) + 1e-12),
                              existing / (np.linalg.norm(existing) + 1e-12), atol=tol):
                    is_dup = True
                    break

            if not is_dup:
                efms_split.append(full_vec)

    # Convert to original coordinates
    efms_orig = []
    for e_split in efms_split:
        e_fwd = e_split[:nR]
        e_bwd = e_split[nR:]
        e_orig = e_fwd - e_bwd

        # Skip if forward and backward both active for same reaction
        if np.any((e_fwd > tol) & (e_bwd > tol)):
            continue

        if np.linalg.norm(e_orig) > tol:
            # Normalize
            e_orig = e_orig / np.linalg.norm(e_orig)
            # Canonical sign
            for x in e_orig:
                if abs(x) > tol:
                    if x < 0:
                        e_orig = -e_orig
                    break

            # Check duplicate
            is_dup = False
            for existing in efms_orig:
                if np.allclose(e_orig, existing, atol=tol):
                    is_dup = True
                    break
            if not is_dup:
                efms_orig.append(e_orig)

    return efms_orig


def enumerate_efms_geometric(Sx: NDArray[np.float64], tol: float = 1e-10) -> list[NDArray[np.float64]]:
    """Enumerate EFMs using geometric extreme-ray enumeration (pycddlib)."""
    try:
        import cdd
    except ImportError:
        raise ImportError("pycddlib is required for geometric method")

    nX, nR = Sx.shape

    # Build split matrix
    Sx_split = np.hstack([Sx, -Sx])
    nR_split = 2 * nR

    # Build H-representation: Sx_split @ v = 0, v >= 0
    # [0 | Sx_split] means Sx_split @ v = 0
    # [0 | -I] means v >= 0

    n_eq = nX
    n_ineq = nR_split

    H = np.zeros((n_eq + n_ineq, 1 + nR_split))
    H[:n_eq, 1:] = Sx_split  # equalities
    H[n_eq:, 0] = 0  # b >= 0 for v >= 0
    H[n_eq:, 1:] = np.eye(nR_split)  # v >= 0

    # Convert to cdd matrix
    mat = cdd.matrix_from_array(H, rep_type=cdd.RepType.INEQUALITY)
    mat.lin_set = frozenset(range(n_eq))  # mark equalities

    poly = cdd.polyhedron_from_matrix(mat)
    generators = cdd.copy_generators(poly)

    efms_split = []
    for row in generators.array:
        if row[0] == 0:  # ray (not vertex)
            ray = np.array(row[1:], dtype=float)
            if np.linalg.norm(ray) > tol:
                efms_split.append(ray)

    # Convert to original coordinates
    efms_orig = []
    for e_split in efms_split:
        e_fwd = e_split[:nR]
        e_bwd = e_split[nR:]
        e_orig = e_fwd - e_bwd

        # Skip if forward and backward both active
        if np.any((e_fwd > tol) & (e_bwd > tol)):
            continue

        if np.linalg.norm(e_orig) > tol:
            e_orig = e_orig / np.linalg.norm(e_orig)
            for x in e_orig:
                if abs(x) > tol:
                    if x < 0:
                        e_orig = -e_orig
                    break

            is_dup = False
            for existing in efms_orig:
                if np.allclose(e_orig, existing, atol=tol):
                    is_dup = True
                    break
            if not is_dup:
                efms_orig.append(e_orig)

    return efms_orig


def run_benchmark(n_values: list[int], timeout: float = 30.0) -> dict:
    """Run benchmark for different network sizes."""
    results = {
        'n': [],
        'n_efms': [],
        'time_combinatorial': [],
        'time_geometric': [],
    }

    for n in n_values:
        print(f"n={n}: ", end="", flush=True)

        Sx = build_modular_assembly_network(n)
        results['n'].append(n)

        # Geometric method
        try:
            t0 = time.perf_counter()
            efms_geo = enumerate_efms_geometric(Sx)
            t_geo = time.perf_counter() - t0
            results['time_geometric'].append(t_geo)
            results['n_efms'].append(len(efms_geo))
            print(f"geo={t_geo:.3f}s ({len(efms_geo)} EFMs), ", end="", flush=True)
        except Exception as e:
            print(f"geo=FAILED ({e}), ", end="", flush=True)
            results['time_geometric'].append(np.nan)
            results['n_efms'].append(np.nan)

        # Combinatorial method (skip if too slow)
        if n <= 6:  # Only run for small n
            try:
                t0 = time.perf_counter()
                efms_comb = enumerate_efms_combinatorial(Sx)
                t_comb = time.perf_counter() - t0
                results['time_combinatorial'].append(t_comb)
                print(f"comb={t_comb:.3f}s ({len(efms_comb)} EFMs)")
            except Exception as e:
                print(f"comb=FAILED ({e})")
                results['time_combinatorial'].append(np.nan)
        else:
            print("comb=SKIPPED (n>6)")
            results['time_combinatorial'].append(np.nan)

    return results


def plot_results(results: dict, outfile: str):
    """Generate fig_EFM_search in both PNG and SVG formats."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    n_vals = np.array(results['n'])
    n_efms = np.array(results['n_efms'])
    t_comb = np.array(results['time_combinatorial'])
    t_geo = np.array(results['time_geometric'])

    # Panel (a): Network schematic (placeholder text)
    ax = axes[0]
    ax.text(0.5, 0.5, "(a) Network schematic\n(see paper Fig. S1)",
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Panel (b): Computation time
    ax = axes[1]

    # Combinatorial
    mask_comb = ~np.isnan(t_comb)
    if np.any(mask_comb):
        ax.semilogy(n_vals[mask_comb], t_comb[mask_comb], 's-',
                   color='#1f77b4', markersize=8, linewidth=2, label='Combinatorial')

    # Geometric
    mask_geo = ~np.isnan(t_geo)
    if np.any(mask_geo):
        ax.semilogy(n_vals[mask_geo], t_geo[mask_geo], 'o-',
                   color='#ff7f0e', markersize=8, linewidth=2, label='Geometric')

    ax.set_xlabel('Target Structure Size (n)', fontsize=12)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('(b)', fontsize=14, fontweight='bold', loc='left')

    # Panel (c): Number of EFMs
    ax = axes[2]
    mask_efm = ~np.isnan(n_efms)
    if np.any(mask_efm):
        ax.semilogy(n_vals[mask_efm], n_efms[mask_efm], 'o-',
                   color='#2ca02c', markersize=8, linewidth=2)

    ax.set_xlabel('Target Structure Size (n)', fontsize=12)
    ax.set_ylabel('Number of EFMs', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('(c)', fontsize=14, fontweight='bold', loc='left')

    plt.tight_layout()

    # Save both PNG and SVG
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Wrote: {outfile}")

    svg_file = outfile.replace('.png', '.svg')
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    print(f"Wrote: {svg_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark EFM enumeration algorithms")
    parser.add_argument('--out', type=str, default='notes/fig_EFM_search.png',
                       help='Output figure path')
    parser.add_argument('--n-max', type=int, default=15,
                       help='Maximum target structure size')
    parser.add_argument('--n-min', type=int, default=3,
                       help='Minimum target structure size')
    args = parser.parse_args()

    n_values = list(range(args.n_min, args.n_max + 1))

    print("Running EFM enumeration benchmark...")
    print(f"Network sizes: n = {args.n_min} to {args.n_max}")
    print()

    results = run_benchmark(n_values)

    print()
    print("Generating figure...")
    plot_results(results, args.out)

    # Also save data
    npz_file = args.out.replace('.png', '.npz')
    np.savez(npz_file, **results)
    print(f"Wrote: {npz_file}")


if __name__ == '__main__':
    main()
