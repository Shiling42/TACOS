"""Verify self-assembly probe EFMs match the paper.

Paper main.tex gives, for extended network [S, probe]:
  e1 = [1, -2, 0, 1]^T
  e2 = [3, 0, 2, 1]^T
  e3 = [0, -3, -1, 1]^T

We enumerate extreme rays on the split irreversible cone and map back.
"""

from __future__ import annotations

import numpy as np

from crn_bounds.stoichiometry import Stoichiometry
from crn_bounds.split import split_reversible
from crn_bounds.efm import enumerate_extreme_rays


def _norm_dir(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    # scale so smallest nonzero abs entry is 1
    nz = np.abs(v[np.abs(v) > 1e-10])
    if nz.size == 0:
        return v
    v = v / np.min(nz)
    # make first nonzero positive
    k = np.argmax(np.abs(v) > 1e-10)
    if v[k] < 0:
        v = -v
    # round
    return np.round(v, 6)


def test_probe_efms_match_paper():
    # full S (X1,X2,X3,F,W)
    S_full = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
        [-1,  0,  0],
        [ 1,  0,  0],
    ], dtype=float)

    # add probe column (only internal part matters for EFMs): [0,-3,2]
    probe = np.array([[0], [-3], [2], [0], [0]], dtype=float)
    S_ext = np.hstack([S_full, probe])
    Sx_ext = S_ext[:3, :]  # internal block (3 x 4)

    split = split_reversible(Stoichiometry(Sx=Sx_ext, Sy=None))
    rays = enumerate_extreme_rays(split.stoich_split.Sx)

    # map back to original 4 reactions
    mapped = []
    for r in rays:
        v = np.zeros(4)
        for k, (rho, sgn) in enumerate(zip(split.split_to_orig, split.split_sign)):
            v[rho] += sgn * r[k]
        if np.linalg.norm(v) > 1e-10:
            mapped.append(_norm_dir(v))

    # expected directions (up to scaling)
    expected = [
        _norm_dir(np.array([1, -2, 0, 1], float)),
        _norm_dir(np.array([3, 0, 2, 1], float)),
        _norm_dir(np.array([0, -3, -1, 1], float)),
    ]

    # each expected should appear
    for e in expected:
        assert any(np.allclose(m, e) for m in mapped), (e, mapped)
