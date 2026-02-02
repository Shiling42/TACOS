"""Test affinity bounds (arXiv:2407.11498).

Key tests:
- Small linear CRN from the paper's Fig. 3 (chemical probe example).
- Correct cycle affinity (e^T A) computation.
- Correct affinity bound (normalized A_e^Y/e_rho) constraints.
- Chemical probe bounds for an effectively computed reaction.
"""

from __future__ import annotations

import numpy as np
import pytest

from crn_bounds.affinity import (
    compute_cycle_affinity,
    compute_affinity_bound,
    compute_probe_bound,
)


def test_paper_figure_3():
    """Test the small linear CRN from paper Fig. 3.

    Layout:
      probe = [1, -1, 0, 0]  # rho_hat (vertical edge)
      A^ss = [0.5, 0.5, -1.0, 0.5]  # steady state affinities

    Testing:
      - For normal reactions: include zero case
      - For probe itself: tighter bounds (skip zero)
    """

    A = np.array([0.5, 0.5, -1.0, 0.5])  # affinities (toy values)

    # EFMs from Fig. 3 ("Green cycles")
    E = [
        # Matches sign convention
        np.array([1, 1, 1, 0]),  # up-left-down
        np.array([0, -1, -1, 1]),  # down-right-up
    ]

    # Test cycle affinities:
    # A_e1 = e1^T A = 0.5 + 0.5 - 1.0 = 0.0
    # A_e2 = e2^T A = (-1)(0.5) + (-1)(-1.0) + (1)(0.5) = -0.5 + 1.0 + 0.5 = 1.0
    assert compute_cycle_affinity(E[0], A) == pytest.approx(0.0)
    assert compute_cycle_affinity(E[1], A) == pytest.approx(1.0)

    # Test affinity bounds for reaction 0:
    # - e0_pos = [1] → A_e/e_0 = 0.0/1 = 0.0
    # - e0_neg = [] → no neg contribution
    # - With zero: min(0, ∅) ≤ A ≤ max(0, 0.0)
    lower, upper = compute_affinity_bound(
        rho=0, A=A, E=E, include_zero=True
    )
    assert lower == pytest.approx(0.0)  # min(0, ∅) = 0
    assert upper == pytest.approx(0.0)  # max(0, 0.0) = 0

    # Same but for chemical probe (no zero-affinity case):
    lower, upper = compute_affinity_bound(
        rho=0, A=A, E=E, include_zero=False
    )
    assert lower == pytest.approx(float("-inf"))  # no neg EFMs
    assert upper == pytest.approx(0.0)  # max ratio of pos EFMs


def test_chemical_probe():
    """Test chemical probe construction (Fig. 3 vertical edge).

    Species layout:
      X1 -- X2   S = [[-1,  0,  1,  0],   # X1
            |     |        [ 1, -1,  0,  1],  # X2
           (ρ)   X3        [ 0,  1, -1,  0],  # X3
            |              [ 0,  0,  0, -1]]  # X4
           X4

    Probe reaction ρ: X4 → X1 (vertical, dotted in the figure)
    Net stoichiometry = [-1, 0, 0, 1] (X4 → X1)

    Note:
      - probe_stoich must be compatible with S (same species order)
      - S describes the original CRN (without the probe)
    """

    # Original CRN stoichiometry (4 species × 4 reactions)
    S = np.array([
        [-1,  0,  1,  0],  # X1
        [ 1, -1,  0,  1],  # X2
        [ 0,  1, -1,  0],  # X3
        [ 0,  0,  0, -1],  # X4
    ])

    # Probe reaction: X4 → X1 (one-unit vertical exchange)
    probe = np.array([-1, 0, 0, 1])  # X1 consumed, X4 produced

    # In reality, the probe bounds come from ALL EFMs in the system
    # because it represents a "virtual infinitesimal" reaction.
    # Let's add more relevant EFMs from Fig. 3:

    # Paper Fig. 3 steady state affinity values:
    # - Half unit for most steps
    # - One unit for the central vertical step
    A = np.array([0.5, 0.5, -1.0, 0.5])  # [horizontal, down, up, right]
    E = [
        # Original cycles from first test:
        np.array([1, 1, 1, 0]),  # left cycle: up-left-down
        np.array([0, -1, -1, 1]),  # right cycle: down-right-up

        # Additional EFMs that define probe bounds:
        np.array([1, 0, -1, 1]),  # diagonal: left-right
        np.array([-1, 1, 1, -1]),  # central square CCW
    ]

    # Test probe bounds:
    lower, upper = compute_probe_bound(probe, S, A, E)

    # Expected: tighter than normal affinity bounds
    # because probe is an "infinitesimally slow" reaction
    assert lower > float("-inf")  # has lower bound
    assert upper < float("inf")   # has upper bound