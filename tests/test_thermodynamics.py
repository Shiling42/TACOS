"""Test thermodynamic space computation.

Key tests:
- Standard Gibbs free energy contributions
- External driving from EFMs (paper Fig. 3)
- Effective equilibrium constant bounds
"""

from __future__ import annotations

import numpy as np
import pytest

from crn_bounds.thermodynamics import (
    compute_effective_constant,
    compute_concentration_bounds,
    visualize_thermo_space,
)


def test_paper_fig3():
    """Test thermodynamic space for paper Fig. 3.
    
    Network layout:
      X1 -- X2   Standard free energies (kJ/mol):
            |     - X1: 0.0 (reference)
           X3     - X2: 2.0
            |     - X3: 1.0
           X4     - X4: -1.0

    Expected:
    - X1 concentration bounded from both sides
    - X2/X3/X4 bounded below (due to cycles)
    """
    # Paper Fig. 3 network:
    S = np.array([
        [-1,  0,  1,  0],  # X1
        [ 1, -1,  0,  1],  # X2
        [ 0,  1, -1,  0],  # X3
        [ 0,  0,  0, -1],  # X4
    ])

    # Standard free energies (kJ/mol)
    dG_0 = np.array([0.0, 2.0, 1.0, -1.0]) * 1000.0  # convert to J/mol

    # External driving (simplified from Fig. 3)
    dG_Y = np.array([-0.5, -0.5, 1.0, -0.5]) * 1000.0  # J/mol

    # EFMs from Fig. 3:
    E = [
        np.array([1, 1, 1, 0]),      # left cycle
        np.array([0, -1, -1, 1]),    # right cycle
        np.array([1, 0, -1, 1]),     # diagonal
        np.array([-1, 1, 1, -1]),    # central square
    ]

    # Test effective equilibrium constant bounds
    # (for X1 → X2 reaction)
    S_rho = np.array([-1, 1, 0, 0])
    K_min, K_max = compute_effective_constant(
        S_rho, dG_0, dG_Y, E,
    )

    # Basic K sanity checks:
    assert K_min > 0.0  # positive (it's exp(-ΔG/RT))
    assert K_max > K_min  # consistent ordering
    assert not np.isinf(K_min)  # finite bounds
    assert not np.isinf(K_max)  # finite bounds

    # Test concentration bounds
    x_min, x_max, species = visualize_thermo_space(
        S, dG_0, dG_Y, E,
        species_names=["X1", "X2", "X3", "X4"],
    )

    # Print the bounds for visual inspection:
    print("\nThermodynamic Space (Fig. 3):")
    print("-" * 40)
    for name, x_lo, x_hi in zip(species, x_min, x_max):
        x_lo_str = f"{x_lo:.2e}" if np.isfinite(x_lo) else "-∞"
        x_hi_str = f"{x_hi:.2e}" if np.isfinite(x_hi) else "∞"
        print(f"{name:>3}: {x_lo_str:>10} ≤ x ≤ {x_hi_str:<10}")
    print("-" * 40)

    # Sanity checks on bounds:
    # 1. All lower bounds should be positive
    assert np.all(x_min > 0)

    # 2. X1 should be bounded on both sides
    assert np.isfinite(x_min[0])
    assert np.isfinite(x_max[0])

    # 3. Other species might be unbounded above
    # (but must be bounded below due to cycles)
    assert np.all(np.isfinite(x_min))  # all lower bounds finite