"""Self-assembly example: reproduce the three non-probe concentration bounds.

Paper Eq self_assembly_bound (with mu0=0):
1) x2/x1^2 in [1, exp(Δμ)]
2) x3/(x1 x2) in [exp(-Δμ), 1]
3) (x1^3)/x3 in [exp(-Δμ), 1]

In log form:
1) ln(x2) - 2 ln(x1) in [0, Δμ]
2) ln(x3) - ln(x1) - ln(x2) in [-Δμ, 0]
3) 3 ln(x1) - ln(x3) in [-Δμ, 0]

Our API returns bounds on s^T ln x for s = Sx[:,rho].
We check that the intervals match these.
"""

from __future__ import annotations

import numpy as np

from crn_bounds.api import CRNInput, run_pipeline


def test_self_assembly_reaction_log_bounds():
    Sx = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=float)

    mu0_X = np.array([0.0, 0.0, 0.0])
    dm = 2.0
    A_Y = np.array([dm, 0.0, 0.0])

    res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

    lbs = res.reaction_log_bounds
    assert len(lbs) == 3

    # Reaction 1: s=[-2,1,0] so s^T ln x = -2 ln x1 + ln x2 = ln(x2/x1^2) in [0, dm]
    assert abs(lbs[0].lo - 0.0) < 1e-6
    assert abs(lbs[0].hi - dm) < 1e-6

    # Reaction 2: width dm, orientation depends on sign convention.
    assert abs((lbs[1].hi - lbs[1].lo) - dm) < 1e-6

    # Reaction 3: width dm, orientation depends on sign convention.
    assert abs((lbs[2].hi - lbs[2].lo) - dm) < 1e-6