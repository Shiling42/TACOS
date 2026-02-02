"""Affinity bounds and chemical probe computations.

Paper: arXiv:2407.11498v2 (Thermodynamic Space of Chemical Reaction Networks).

Core concept: local reaction affinities (at steady state) are bounded by
cycle affinities along Elementary Flux Modes (EFMs), normalized by their
participation counts in each extreme mode.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Sequence

from .utils import FLOAT_TOL, compute_efm_affinity_ratios


def compute_cycle_affinity(
    e: NDArray[np.float64],  # (nR,) EFM vector
    A: NDArray[np.float64],  # (nR,) affinity vector
    *,
    internal: bool = False,   # if True, include internal A^X components
) -> float:
    """Cycle affinity A_e for an EFM/extreme ray.
    
    Paper Eq. (8):
        A_e = e^T A = e^T A^X + e^T A^Y = A^Y_e

    Args:
        e: EFM vector (nR,)
        A: affinity vector (nR,)
        internal: if True, include A^X internal affinity

    Returns:
        float: cycle affinity A^Y_e (by default: external only)
    """
    return float(e @ A)  # e^T A = sum(e_i A_i)


def compute_probe_bound(
    probe_stoich: NDArray[np.float64],   # (nS,) probe stoichiometry vector
    S: NDArray[np.float64],              # (nS, nR) stoichiometry matrix
    A: NDArray[np.float64],              # (nR,) affinity vector
    E: Sequence[NDArray[np.float64]],    # list of EFM vectors
) -> tuple[float, float]:
    """Paper Eq. (24): Chemical probe affinity bounds.
    
    Args:
        probe_stoich: stoichiometry vector of the probe reaction (nS,)
        S: stoichiometry matrix of the original CRN (nS, nR)
        A: affinity vector of original reactions (nR,)
        E: list of Elementary Flux Modes [(nR,), ...]
    
    Returns:
        (lower, upper) bounds on steady-state probe affinity.
        These are tighter than Eq. (20) due to probe construction.
    """

    # Step 1: Project probe onto the reaction space
    # (this is what the paper does implicitly by extending S)
    S_extended = np.hstack([S, probe_stoich.reshape(-1, 1)])
    n_reactions = S.shape[1]

    pos_ratios: list[float] = []  # A_e^Y/e_rho for e_rho > 0
    neg_ratios: list[float] = []  # A_e^Y/e_rho for e_rho < 0

    for e in E:
        # We need to "lift" e to include probe participation:
        # - Original reactions: use e as is
        # - Probe reaction: project via stoichiometry
        probe_coeff = float(probe_stoich.T @ (S @ e))
        if abs(probe_coeff) < FLOAT_TOL:  # probe not involved in this cycle
            continue

        A_e = compute_cycle_affinity(e, A, internal=False)
        ratio = A_e / probe_coeff  # normalize by probe participation

        if probe_coeff > 0:
            pos_ratios.append(ratio)
        else:  # probe_coeff < 0
            neg_ratios.append(ratio)

    # Chemical probe bounds are strict (no zero-affinity case)
    return (
        min(neg_ratios) if neg_ratios else float("-inf"),
        max(pos_ratios) if pos_ratios else float("inf"),
    )


def compute_affinity_bound(
    rho: int,
    A: NDArray[np.float64],          # (nR,) affinity vector
    E: Sequence[NDArray[np.float64]], # list of EFM vectors
    include_zero: bool = False,       # set True for paper Eq. (20)
) -> tuple[float, float]:
    """Paper Eq. (20): Local affinity bounds from EFM affinities.

    Args:
        rho: target reaction index
        A: affinity vector (nR,)
        E: list of EFM/extreme ray vectors [(nR,), ...]
        include_zero: if True, enforce full Eq. (20) bounds
            with zero-affinity case.

    Returns:
        (lower, upper) bounds on steady-state affinity.
    """
    pos_ratios, neg_ratios = compute_efm_affinity_ratios(rho, A, list(E))

    if not pos_ratios and not neg_ratios:
        # No EFMs contain rho: zero affinity (detailed balance)
        return (0.0, 0.0)

    if include_zero:
        # Full Eq. (20) from paper: min(0, min_neg) ≤ A ≤ max(0, max_pos)
        neg_bound = min(neg_ratios) if neg_ratios else 0.0
        pos_bound = max(pos_ratios) if pos_ratios else 0.0
        return (min(0.0, neg_bound), max(0.0, pos_bound))

    else:
        # Simple bounds from probe-like situation: no zero-affinity case
        # (used in the paper's Eq. (24) for chemical probe construction)
        return (
            min(neg_ratios) if neg_ratios else float("-inf"),
            max(pos_ratios) if pos_ratios else float("inf"),
        )