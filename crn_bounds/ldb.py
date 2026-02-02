"""Local detailed balance (LDB) utilities.

We provide a way to sample mass-action kinetic rates consistent with a given
thermodynamic assignment:
- internal standard chemical potentials mu0_X (dimensionless: mu/RT)
- external chemostat chemical potentials mu_Y (dimensionless: mu/RT)

For each reversible reaction rho with stoichiometry (nu_plus -> nu_minus):
  ln(k_plus/k_minus) = - (Δmu0_X + Δmu_Y)
where
  Δmu0_X = sum_i (nu_minus - nu_plus)_i * mu0_X[i]
  Δmu_Y  = sum_j (nu_minus - nu_plus)_j * mu_Y[j]

This matches LDB at standard state / with chemostats.

We randomize a symmetric prefactor a_rho and set:
  k_plus  = a_rho * exp(+ell_rho/2)
  k_minus = a_rho * exp(-ell_rho/2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .model import MassActionCRN


def ldb_log_ratio(
    crn: MassActionCRN,
    mu0_X: NDArray[np.float64],
    mu_Y: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Compute ell_rho = ln(k+/k-) for each reaction rho."""
    Sx = crn.Sx()  # nu_minus - nu_plus
    ell = -(Sx.T @ mu0_X)

    if crn.nuY_plus is not None:
        if mu_Y is None:
            raise ValueError("mu_Y must be provided when CRN has external species")
        Sy = crn.nuY_minus - crn.nuY_plus
        ell = ell - (Sy.T @ mu_Y)

    return ell


def sample_rates_from_ldb(
    crn: MassActionCRN,
    mu0_X: NDArray[np.float64],
    mu_Y: NDArray[np.float64] | None,
    *,
    rng: np.random.Generator,
    loga_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample (k_plus,k_minus) consistent with LDB.

    Args:
      mu0_X: internal standard chemical potentials (dimensionless mu/RT)
      mu_Y: external chemical potentials (dimensionless mu/RT)
      loga_range: range for log(a_rho) prefactors

    Returns:
      k_plus, k_minus arrays (nR,)
    """
    nR = crn.nuX_plus.shape[1]
    ell = ldb_log_ratio(crn, mu0_X, mu_Y)
    loga = rng.uniform(loga_range[0], loga_range[1], size=nR)
    a = np.exp(loga)

    k_plus = a * np.exp(0.5 * ell)
    k_minus = a * np.exp(-0.5 * ell)

    return k_plus, k_minus
