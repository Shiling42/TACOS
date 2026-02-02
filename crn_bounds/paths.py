"""Pathway-based bounds (Pi^X sets) from EFMs.

Implements the paper's construction around Eq. (24)-(28):
- Given a target reaction rho (original network), define pathway affinities
  after eliminating rho from EFMs that contain rho.

For an EFM e with e_rho > 0:
  pathway affinity converting X from right -> left of rho is:
    A_pi^Y = A_e^Y / e_rho - A_rho^Y
(see text below Eq. (24) in main.tex).

Additionally, Pi^X_{rho^-} includes the direct backward reaction with affinity -A_rho^Y.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PathwayBounds:
    lower: float
    upper: float


def pathway_affinities_from_efms(
    rho: int,
    E: Sequence[NDArray[np.float64]],
    A_Y: NDArray[np.float64],
) -> tuple[list[float], list[float]]:
    """Compute pathway affinity samples for Pi_{rho^-} and Pi_{rho^+}.

    Returns:
      (pi_minus_affinities, pi_plus_affinities)

    Convention:
      - pi_minus: converts X from right to left of reaction rho
      - pi_plus: converts X from left to right of reaction rho

    For e_rho>0 we get a pi_minus pathway (right->left) from eliminating rho.
    For e_rho<0 we get a pi_plus pathway (left->right).
    """
    A_rho_Y = float(A_Y[rho])
    pi_minus = [-A_rho_Y]  # direct backward reaction
    pi_plus = [A_rho_Y]    # direct forward reaction

    for e in E:
        e_rho = float(e[rho])
        if abs(e_rho) < 1e-12:
            continue
        A_e_Y = float(e @ A_Y)
        if e_rho > 0:
            pi_minus.append(A_e_Y / e_rho - A_rho_Y)
        else:
            # eliminate rho but now conversion is opposite direction
            # Use symmetry: pi_plus from e_rho<0
            pi_plus.append(A_e_Y / e_rho - A_rho_Y)

    return pi_minus, pi_plus


def bounds_from_pathway_affinities(
    pi: Sequence[float],
) -> PathwayBounds:
    return PathwayBounds(lower=float(min(pi)), upper=float(max(pi)))
