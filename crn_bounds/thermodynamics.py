"""Thermodynamic space computation (concentration bounds).

Paper: arXiv:2407.11498v2 (Sec. IV.B: Thermodynamic Space).

Core concept: from affinity bounds, we can derive bounds on the
steady-state concentrations of internal species.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Sequence
from scipy.optimize import linprog


def compute_effective_constant(
    S_rho: NDArray[np.float64],   # (nS,) stoichiometry vector
    dG_0: NDArray[np.float64],    # (nS,) standard free energies
    dG_Y: NDArray[np.float64],    # (nR,) external driving terms
    E: Sequence[NDArray[np.float64]], # EFMs from split CRN
    *,
    R: float = 8.314462618,  # gas constant (J/mol·K)
    T: float = 298.15,       # temperature (K)
) -> tuple[float, float]:
    """Paper Eq. (28): Effective equilibrium constant bounds.
    
    Args:
        S_rho: stoichiometry vector of target reaction
        dG_0: standard free energy changes (internal)
        dG_Y: external driving terms
        E: Elementary Flux Modes from split CRN
        R: gas constant (J/mol·K)
        T: temperature (K)

    Returns:
        (K_min, K_max): bounds on effective equilibrium constant
    """
    RT = R * T  # thermal energy scale

    # Step 1: Standard contribution (internal species)
    dG_X = float(S_rho @ dG_0)  # dot product: internal free energy

    # Step 2: Find extremal contributions from EFMs
    min_dG_Y = float("inf")
    max_dG_Y = float("-inf")

    for e in E:
        dG_e = float(e @ dG_Y)  # external work along this EFM
        if dG_e < min_dG_Y:
            min_dG_Y = dG_e
        if dG_e > max_dG_Y:
            max_dG_Y = dG_e

    # Step 3: Compute effective equilibrium constants
    # (using the extremal external contributions)
    K_min = np.exp(-(dG_X + max_dG_Y) / RT)
    K_max = np.exp(-(dG_X + min_dG_Y) / RT)

    return K_min, K_max


def compute_concentration_bounds(
    S: NDArray[np.float64],    # (nS, nR) stoichiometry matrix
    dG_0: NDArray[np.float64], # (nS,) standard free energies
    dG_Y: NDArray[np.float64], # (nR,) external driving
    E: Sequence[NDArray[np.float64]],  # EFMs
    *,
    R: float = 8.314462618,  # gas constant (J/mol·K)
    T: float = 298.15,       # temperature (K)
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute bounds on steady-state concentrations.
    
    The problem is formulated as a set of linear constraints on log(x):
        ln(K_min) ≤ ∑_i S_i ln(x_i) ≤ ln(K_max)
    
    For each species i, we solve two LPs to get min/max ln(x_i).
    
    Args:
        S: stoichiometry matrix (internal species only)
        dG_0: standard free energies of internal species
        dG_Y: external driving terms
        E: Elementary Flux Modes from split CRN
        R: gas constant (J/mol·K)
        T: temperature (K)

    Returns:
        (x_min, x_max): component-wise bounds on internal
        species concentrations (in mol/L).
    """
    n_species, n_reactions = S.shape

    # Build constraints from each reaction's K bounds
    A_ub: list[NDArray[np.float64]] = []  # A_ub @ x ≤ b_ub
    b_ub: list[float] = []

    for rho in range(n_reactions):
        S_rho = S[:, rho]  # stoichiometry vector
        if np.all(S_rho == 0):  # no internal species
            continue

        # Get K bounds for this reaction
        K_min, K_max = compute_effective_constant(
            S_rho, dG_0, dG_Y, E, R=R, T=T
        )

        # Add two constraints per reaction:
        # 1) ∑ S_i ln(x_i) ≥ ln(K_min)  →  -∑ S_i ln(x_i) ≤ -ln(K_min)
        # 2) ∑ S_i ln(x_i) ≤ ln(K_max)
        A_ub.extend([-S_rho, S_rho])
        b_ub.extend([-np.log(K_min), np.log(K_max)])

    # Convert to arrays for linprog
    A_ub = np.vstack(A_ub) if A_ub else np.zeros((0, n_species))
    b_ub = np.array(b_ub) if b_ub else np.zeros(0)

    # For each species: solve LP to get min/max ln(x_i)
    ln_x_min = np.zeros(n_species)
    ln_x_max = np.zeros(n_species)

    for i in range(n_species):
        # Minimize x_i: min e_i^T ln(x)
        c = np.zeros(n_species)
        c[i] = 1.0
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            bounds=(None, None),  # ln(x) unbounded
            method='highs',
        )
        ln_x_min[i] = res.x[i] if res.success else -10.0  # ~1e-6

        # Maximize x_i: max e_i^T ln(x) = min (-e_i^T ln(x))
        res = linprog(
            -c, A_ub=A_ub, b_ub=b_ub,
            bounds=(None, None),
            method='highs',
        )
        ln_x_max[i] = res.x[i] if res.success else 100.0  # ~1e43

    # Check for feasibility:
    if not A_ub.size:  # no constraints
        return (
            np.full(n_species, 1e-6),  # arbitrary small positive
            np.full(n_species, np.inf)
        )

    # Test if the constraints are feasible:
    res = linprog(
        np.zeros(n_species),  # any objective
        A_ub=A_ub, b_ub=b_ub,
        bounds=(None, None),
        method='highs',
    )
    if not res.success:  # infeasible
        return (
            np.full(n_species, np.nan),
            np.full(n_species, np.nan)
        )

    # Convert back from log space:
    # - Finite positive lower bounds
    # - Allow infinite upper bounds
    x_min = np.maximum(np.exp(ln_x_min), 1e-6)
    x_max = np.where(
        np.isfinite(ln_x_max),
        np.exp(ln_x_max),
        np.inf
    )

    return x_min, x_max


def visualize_thermo_space(
    S: NDArray[np.float64],
    dG_0: NDArray[np.float64],
    dG_Y: NDArray[np.float64],
    E: Sequence[NDArray[np.float64]],
    species_names: Sequence[str] | None = None,
    *,
    R: float = 8.314462618,
    T: float = 298.15,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Compute and visualize the thermodynamic space.
    
    Args:
        S: stoichiometry matrix (nS × nR)
        dG_0: standard free energies (nS,)
        dG_Y: external driving terms (nR,)
        E: list of EFM vectors [(nR,), ...]
        species_names: optional list of species names

    Returns:
        x_min, x_max: concentration bounds arrays
        names: list of species names (or X1, X2, etc.)
    """
    n_species = S.shape[0]
    if species_names is None:
        species_names = [f"X{i+1}" for i in range(n_species)]

    # Compute concentration bounds
    x_min, x_max = compute_concentration_bounds(
        S, dG_0, dG_Y, E, R=R, T=T
    )

    return x_min, x_max, list(species_names)