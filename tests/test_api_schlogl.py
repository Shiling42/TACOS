"""Schlogl model example from paper: verify affinity bounds + concentration bounds.

Paper Eq. con_bound_Schlogl:
  exp(-(mu_X^0 - mu_B)) <= x <= exp(-(mu_X^0 - mu_A))
With mu0_X (internal) and external chemical potentials embedded in A_Y.

Here we test affinity bounds only with A_Y and EFM.
"""

from __future__ import annotations

import numpy as np

from crn_bounds.api import CRNInput, run_pipeline


def test_schlogl_affinity_bounds():
    # Sx: internal species X, 2 reactions
    # reaction1: 2X + A <-> 3X  => net +1 X
    # reaction2: X <-> B       => net -1 X (for internal X)
    Sx = np.array([[1, -1]], dtype=float)

    mu0_X = np.array([0.0])

    # Driving along EFM converts A -> B with affinity mu_A - mu_B.
    dm = 1.7
    A_Y = np.array([dm, 0.0])  # assign all driving to reaction 1 for test

    res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y))

    # Both reactions should have 0<=A<=dm
    for b in res.affinity_bounds:
        assert b.lower <= 1e-9
        assert abs(b.upper - dm) < 1e-6
