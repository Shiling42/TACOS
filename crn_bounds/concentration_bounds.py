"""Concentration (log-ratio) bounds for reactions/probes.

We express thermodynamic concentration constraints as bounds on linear forms of log-concentrations:

  lo <= s^T ln x <= hi

where s is an internal stoichiometry vector for an interconversion.

This module provides:
- probe_log_ratio_bound(): bounds for an *added probe* reaction from extended-network EFMs.
- reaction_log_ratio_bound(): bounds for an *existing reaction* from EFMs of the original network.

Theory (paper arXiv:2407.11498, Π-pathway construction):

For a reaction ρ with internal stoichiometry s = Sx[:,ρ]:
- For each EFM e with e_ρ ≠ 0 define per-conversion external affinity: a = (e^T A_Y)/e_ρ
- Then a_min ≤ A_ρ^{ss} ≤ a_max (paper Eq. 20 with normalization)
- Translate to log constraint via:
    A_ρ^{X,ss} = -ΔG_X^0 - s^T ln x
    A_ρ^{ss} = A_ρ^{X,ss} + A_ρ^Y

With A_ρ^Y = A_Y[ρ], we have:
    s^T ln x = -ΔG_X^0 - (A_ρ^{ss} - A_ρ^Y)

So with A_ρ^{ss} ∈ [a_min, a_max]:
    lo = -ΔG^0 - (a_max - A_ρ^Y)
    hi = -ΔG^0 - (a_min - A_ρ^Y)

For pure internal probes, A_probe^Y = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LogRatioBound:
    s: NDArray[np.float64]  # (nX,)
    lo: float
    hi: float


def _canonical_sign_lastpos(s: NDArray[np.float64], lo: float, hi: float) -> tuple[NDArray[np.float64], float, float]:
    """Flip sign so that the last nonzero entry is positive, adjusting interval."""
    s0 = s.copy()
    for x in reversed(s0):
        if abs(x) > 1e-12:
            if x < 0:
                s0 = -s0
                lo, hi = -hi, -lo
            break
    if lo > hi:
        lo, hi = hi, lo
    return s0, float(lo), float(hi)


def probe_log_ratio_bound(
    *,
    s_probe: NDArray[np.float64],
    mu0_X: NDArray[np.float64],
    A_Y_ext: NDArray[np.float64],
    efms_ext: Sequence[NDArray[np.float64]],
    probe_index: int,
) -> LogRatioBound:
    """Compute log-ratio bound for an internal probe reaction."""
    s = np.asarray(s_probe, dtype=float)
    dG0 = float(s @ mu0_X)  # ΔG_X^0/RT

    ratios = []
    for e in efms_ext:
        e_p = float(e[probe_index])
        if abs(e_p) < 1e-12:
            continue
        A_e = float(e @ A_Y_ext)
        ratios.append(A_e / e_p)

    if not ratios:
        return LogRatioBound(s=s, lo=float("-inf"), hi=float("inf"))

    a_min = float(min(ratios))
    a_max = float(max(ratios))

    lo = -dG0 - a_max
    hi = -dG0 - a_min

    s0, lo, hi = _canonical_sign_lastpos(s, lo, hi)
    return LogRatioBound(s=s0, lo=lo, hi=hi)


def reaction_log_ratio_bound(
    *,
    rho: int,
    Sx: NDArray[np.float64],
    mu0_X: NDArray[np.float64],
    A_Y: NDArray[np.float64],
    efms: Sequence[NDArray[np.float64]],
    include_zero: bool = True,
) -> LogRatioBound:
    """Compute reaction concentration bound using general Π-pathway theory.

    For a reaction ρ with internal stoichiometry s = Sx[:,ρ]:
    - At steady state: A_ρ^ss = A_ρ^X + A_ρ^Y
    - Where A_ρ^X = -ΔG⁰ - s^T ln x (internal affinity from concentrations)
    - And A_ρ^Y = A_Y[ρ] (external driving)

    From EFM-based affinity bounds (paper Eq. 20, with include_zero=True):
    - For each EFM e with e_ρ ≠ 0: a = (e^T A_Y) / e_ρ
    - Positive ratios give upper bounds, negative ratios give lower bounds
    - With include_zero=True: a_min = min(0, min_neg), a_max = max(0, max_pos)

    Translating to concentration bounds:
    - A_ρ^ss = -ΔG⁰ - s^T ln x + A_ρ^Y
    - a_min ≤ -ΔG⁰ - s^T ln x + A_ρ^Y ≤ a_max
    - Therefore: lo = -ΔG⁰ - (a_max - A_ρ^Y), hi = -ΔG⁰ - (a_min - A_ρ^Y)

    This is the general formula that works for arbitrary CRNs.
    """
    s = np.asarray(Sx[:, rho], dtype=float)
    dG0 = float(s @ mu0_X)
    A_rho_Y = float(A_Y[rho])  # external affinity for this reaction

    if not efms:
        return LogRatioBound(s=s, lo=float("-inf"), hi=float("inf"))

    # Compute per-reaction normalized cycle affinities: a = (e^T A_Y) / e_ρ
    # Separate positive and negative EFM participation (like affinity.py)
    pos_ratios: list[float] = []
    neg_ratios: list[float] = []

    for e in efms:
        e_r = float(e[rho])
        if abs(e_r) < 1e-12:
            continue
        ratio = float((e @ A_Y) / e_r)
        if e_r > 0:
            pos_ratios.append(ratio)
        else:
            neg_ratios.append(ratio)

    if not pos_ratios and not neg_ratios:
        # Reaction not in any EFM: at detailed balance (A_ρ^ss = 0)
        # s^T ln x = -ΔG⁰ - (0 - A_ρ^Y) = -ΔG⁰ + A_ρ^Y
        eq_val = -dG0 + A_rho_Y
        s0, lo, hi = _canonical_sign_lastpos(s, eq_val, eq_val)
        return LogRatioBound(s=s0, lo=lo, hi=hi)

    # Compute affinity bounds following paper Eq. (20) with include_zero
    if include_zero:
        # Full Eq. (20): min(0, min_neg) ≤ A ≤ max(0, max_pos)
        neg_bound = min(neg_ratios) if neg_ratios else 0.0
        pos_bound = max(pos_ratios) if pos_ratios else 0.0
        a_min = min(0.0, neg_bound)
        a_max = max(0.0, pos_bound)
    else:
        # Strict bounds (for probes): no zero-affinity case
        all_ratios = pos_ratios + neg_ratios
        a_min = min(all_ratios)
        a_max = max(all_ratios)

    # Translate affinity bounds to concentration bounds:
    # lo = -ΔG⁰ - (a_max - A_ρ^Y)
    # hi = -ΔG⁰ - (a_min - A_ρ^Y)
    lo = -dG0 - (a_max - A_rho_Y)
    hi = -dG0 - (a_min - A_rho_Y)

    # canonicalize sign using last-nonzero-positive convention
    s0, lo, hi = _canonical_sign_lastpos(s, lo, hi)
    return LogRatioBound(s=s0, lo=lo, hi=hi)
