"""Test auto_probes integration on self-assembly.

We expect probe candidate [0,-3,2] and probe affinity bound [-3Δμ, 0]
from paper (A_hat in [-3Δμ, 0]).

In our A_Y input convention (dimensionless A^Y/RT): A_Y=[Δμ,0,0].
"""

from __future__ import annotations

import numpy as np

from crn_bounds.api import CRNInput, run_pipeline


def test_self_assembly_auto_probe():
    Sx = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=float)

    mu0_X = np.array([0.0, 0.0, 0.0])
    dm = 2.0
    A_Y = np.array([dm, 0.0, 0.0])

    res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=True, max_probes=10)

    assert len(res.probes) >= 1

    # find target probe
    target = np.array([0, -3, 2])

    def norm(v):
        v = v.astype(int)
        g = np.gcd.reduce(np.abs(v[v != 0])) if np.any(v != 0) else 1
        v = v // g
        for x in v:
            if x != 0:
                if x < 0:
                    v = -v
                break
        return v

    t = norm(target)

    hit = None
    for pr in res.probes:
        if np.allclose(norm(pr.probe_sx), t):
            hit = pr
            break

    assert hit is not None

    # probe affinity bound should be [-3*dm, 0]
    assert abs(hit.affinity_bound.upper - 0.0) < 1e-6
    assert abs(hit.affinity_bound.lower - (-3 * dm)) < 1e-6

    # probe concentration (log-ratio) bound on s^T ln x.
    # For s=[0,-3,2] and mu0=0, paper gives:
    #   ln((x3^2)/(x2^3)) in [-3*dm, 0]
    # Our LogRatioBound is for s^T ln x = 2 ln x3 - 3 ln x2
    assert abs(hit.log_ratio_bound.hi - 0.0) < 1e-6
    assert abs(hit.log_ratio_bound.lo - (-3 * dm)) < 1e-6
