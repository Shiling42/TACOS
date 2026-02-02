"""Frank (chiral symmetry breaking) example: verify probe log-ratio bound.

From paper:
- Internal species: R,S
- External chemostats: A,C (but we encode driving via A_Y)
- Original reactions (3):
  1) R + A <-> 2R
  2) S + A <-> 2S
  3) R + S <-> C
Sx internal block (R,S):
  col1: [ +1,  0]
  col2: [  0, +1]
  col3: [ -1, -1]

Probe: S <-> R has s=[1,-1]
Paper bound: ln(r/s) in [-Δμ, +Δμ] (dimensionless /RT)

We test that our probe log_ratio_bound reproduces this when we provide the probe explicitly.
"""

from __future__ import annotations

import numpy as np

from crn_bounds.stoichiometry import Stoichiometry
from crn_bounds.split import split_reversible
from crn_bounds.efm import enumerate_extreme_rays
from crn_bounds.concentration_bounds import probe_log_ratio_bound


def test_frank_probe_log_ratio_bound():
    # internal block only (R,S)
    Sx = np.array([
        [ 1, 0, -1],
        [ 0, 1, -1],
    ], dtype=float)

    dm = 2.5  # Δμ/RT

    # Encode all driving on the single EFM e=[1,1,1] by assigning A_Y=[dm, dm, -dm]? too detailed.
    # We only need that along extended EFMs the max normalized external affinity equals +dm and min equals -dm.
    # Use a simple assignment: A_Y for reactions 1 and 2 carry +dm, reaction 3 carries -dm so that e·A_Y = dm.
    A_Y = np.array([dm, dm, -dm], dtype=float)

    # add probe as reaction 4 in internal space: s=[1,-1]
    probe = np.array([[1], [-1]], dtype=float)
    Sx_ext = np.hstack([Sx, probe])
    A_Y_ext = np.concatenate([A_Y, [0.0]])

    split = split_reversible(Stoichiometry(Sx=Sx_ext, Sy=None))
    rays = enumerate_extreme_rays(split.stoich_split.Sx)

    # map to original+probe
    efms = []
    for r in rays:
        v = np.zeros(4)
        for k, (rho, sgn) in enumerate(zip(split.split_to_orig, split.split_sign)):
            v[rho] += sgn * r[k]
        if np.linalg.norm(v) > 1e-12:
            efms.append(v)

    lb = probe_log_ratio_bound(
        s_probe=np.array([1.0, -1.0]),
        mu0_X=np.array([0.0, 0.0]),
        A_Y_ext=A_Y_ext,
        efms_ext=efms,
        probe_index=3,
    )

    # bound on s^T ln x = ln(r) - ln(s) = ln(r/s)
    assert abs(lb.hi - dm) < 1e-6
    assert abs(lb.lo - (-dm)) < 1e-6