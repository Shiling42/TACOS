"""Thermodynamic-space constraints derived from pathway affinity bounds.

Paper link:
- Eq. (26): affinity/free-energy decomposition
- Eq. (28)/(eq_constant_bound_0): effective equilibrium constants for pathways

We represent constraints in log-space:
  l_rho <= s_rho^T ln x <= u_rho
where s_rho is a stoichiometric vector (internal species) for an interconversion.

Given pathway free-energy bounds (external part) we can compute l/u.

This module focuses on producing constraints; plotting / projection is separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .paths import pathway_affinities_from_efms, bounds_from_pathway_affinities


@dataclass(frozen=True)
class LogConstraint:
    s: NDArray[np.float64]  # (nX,)
    lo: float               # lower bound on s^T ln x
    hi: float               # upper bound on s^T ln x
    name: str


def reaction_log_constraint(
    rho: int,
    Sx: NDArray[np.float64],
    mu0: NDArray[np.float64],
    A_Y: NDArray[np.float64],
    E: Sequence[NDArray[np.float64]],
    *,
    RT: float = 1.0,
    name: str | None = None,
) -> LogConstraint:
    """Compute the log-concentration constraint for reaction rho.

    Using Eq. affinity_bound_2 + affinity_free_energy.

    We compute pathway bounds for Pi_{rho^-} and Pi_{rho^+} from EFMs.
    Then translate to bounds on s^T ln x.

    Derivation sketch:
      A_rho^{X,ss} = -ΔG_X° - RT * (s^T ln x)
      And A_rho^{X,ss} is bounded by pathway affinities (external):
        min_{Pi_{rho^+}} A_pi^Y <= A_rho^{X,ss} <= max_{Pi_{rho^-}} A_pi^Y

      Solve for s^T ln x:
        (-ΔG_X° - upper)/RT <= s^T ln x <= (-ΔG_X° - lower)/RT

    (Signs depend on conventions; we keep consistency with self-assembly formulas.)
    """

    s = np.asarray(Sx[:, rho], dtype=float)
    dG_X0 = float(s @ mu0)  # ΔG_X° = sum_i S_i μ_i°

    pi_minus, pi_plus = pathway_affinities_from_efms(rho, E, A_Y)
    b_minus = bounds_from_pathway_affinities(pi_minus)  # bounds on A_pi^Y for rho^- pathways
    b_plus = bounds_from_pathway_affinities(pi_plus)

    # A_rho^{X,ss} ∈ [min Pi_plus, max Pi_minus]
    A_X_lo = b_plus.lower
    A_X_hi = b_minus.upper

    lo = (-dG_X0 - A_X_hi) / RT
    hi = (-dG_X0 - A_X_lo) / RT

    if name is None:
        name = f"rho{rho}"

    return LogConstraint(s=s, lo=float(lo), hi=float(hi), name=name)
