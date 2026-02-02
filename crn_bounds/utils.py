"""Shared utilities for CRN bounds computations.

This module provides common functions used across the package:
- Tolerance constant for floating point comparisons
- Vector canonicalization (sign normalization, primitive form)
- EFM affinity ratio computation
- EFM coordinate mapping
"""

from __future__ import annotations

from functools import reduce
from math import gcd

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Tolerance constant for floating point comparisons
# =============================================================================
FLOAT_TOL = 1e-12


# =============================================================================
# Helper functions for integer operations
# =============================================================================
def _gcd_list(xs: list[int]) -> int:
    """Compute GCD of a list of integers."""
    xs = [abs(x) for x in xs if x != 0]
    return reduce(gcd, xs, 0) if xs else 1


# =============================================================================
# Vector canonicalization
# =============================================================================
def canonicalize_vector(
    v: NDArray[np.float64],
    *,
    make_primitive: bool = False,
    tol: float = FLOAT_TOL,
) -> NDArray[np.float64]:
    """Canonicalize a vector so first nonzero entry is positive.

    Args:
        v: Input vector
        make_primitive: If True, divide by GCD (for integer vectors)
        tol: Tolerance for zero comparison

    Returns:
        Canonicalized vector (copy)
    """
    v = v.copy()

    if make_primitive:
        # For integer vectors: divide by GCD
        nonzero = v[np.abs(v) > tol]
        if len(nonzero) > 0:
            g = _gcd_list([int(round(x)) for x in nonzero])
            if g > 1:
                v = v / g

    # Make first nonzero positive
    for x in v:
        if abs(x) > tol:
            if x < 0:
                v = -v
            break

    return v


def canonicalize_with_bounds(
    s: NDArray[np.float64],
    lo: float,
    hi: float,
    *,
    tol: float = FLOAT_TOL,
) -> tuple[NDArray[np.float64], float, float]:
    """Canonicalize vector and flip bounds accordingly (last-nonzero-positive convention).

    Used for log-ratio constraints where we want the last nonzero entry positive.

    Args:
        s: Stoichiometry vector
        lo: Lower bound
        hi: Upper bound
        tol: Tolerance for zero comparison

    Returns:
        (s_canonical, lo_canonical, hi_canonical)
    """
    s0 = s.copy()
    for x in reversed(s0):
        if abs(x) > tol:
            if x < 0:
                s0 = -s0
                lo, hi = -hi, -lo
            break
    if lo > hi:
        lo, hi = hi, lo
    return s0, float(lo), float(hi)


# =============================================================================
# EFM affinity ratio computation
# =============================================================================
def compute_efm_affinity_ratios(
    reaction_index: int,
    A_Y: NDArray[np.float64],
    efms: list[NDArray[np.float64]],
    *,
    tol: float = FLOAT_TOL,
) -> tuple[list[float], list[float]]:
    """Compute affinity ratios (e^T A_Y) / e_rho for positive and negative e_rho.

    This is the core EFM iteration pattern from paper Eq. (20).

    Args:
        reaction_index: Index rho of the target reaction
        A_Y: External affinity vector (nR,)
        efms: List of EFM vectors [(nR,), ...]
        tol: Tolerance for zero comparison

    Returns:
        (pos_ratios, neg_ratios): Lists of A_e/e_rho for e_rho > 0 and e_rho < 0
    """
    pos_ratios: list[float] = []
    neg_ratios: list[float] = []

    for e in efms:
        e_rho = float(e[reaction_index])
        if abs(e_rho) < tol:  # Use tolerance instead of exact zero
            continue

        A_e = float(e @ A_Y)
        ratio = A_e / e_rho

        if e_rho > 0:
            pos_ratios.append(ratio)
        else:
            neg_ratios.append(ratio)

    return pos_ratios, neg_ratios


# =============================================================================
# EFM coordinate mapping
# =============================================================================
def map_split_efms_to_original(
    efms_split: list[NDArray[np.float64]],
    split_to_orig: list[int],
    split_sign: list[int],
    n_reactions_orig: int,
    *,
    tol: float = FLOAT_TOL,
    canonicalize: bool = True,
    deduplicate: bool = True,
) -> list[NDArray[np.float64]]:
    """Map EFMs from split coordinates back to original reaction coordinates.

    Args:
        efms_split: EFMs in split representation
        split_to_orig: Mapping from split index to original reaction index
        split_sign: Sign (+1/-1) for each split reaction
        n_reactions_orig: Number of reactions in original network
        tol: Tolerance for deduplication
        canonicalize: If True, make first nonzero positive
        deduplicate: If True, remove duplicate EFMs

    Returns:
        List of EFMs in original coordinates
    """
    efms_orig: list[NDArray[np.float64]] = []

    for e in efms_split:
        v = np.zeros(n_reactions_orig)
        for k, (rho, sgn) in enumerate(zip(split_to_orig, split_sign)):
            v[rho] += sgn * e[k]

        if np.linalg.norm(v) < tol:
            continue

        if canonicalize:
            v = canonicalize_vector(v, tol=tol)

        if deduplicate:
            if any(np.allclose(v, u, atol=tol) for u in efms_orig):
                continue

        efms_orig.append(v)

    return efms_orig
