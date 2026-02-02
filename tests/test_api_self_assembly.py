"""Test the standardized API on the paper self-assembly network."""

from __future__ import annotations

import numpy as np

from crn_bounds.api import CRNInput, run_pipeline


def test_self_assembly_affinity_bounds():
    # Sx from paper Eq self_assembly_reac (internal block)
    Sx = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=float)

    # caption: mu0 all zero
    mu0_X = np.array([0.0, 0.0, 0.0])

    # only reaction 1 carries external driving: A_Y = [Δμ,0,0]
    dm = 2.0
    A_Y = np.array([dm, 0.0, 0.0])

    res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y))

    # should have an EFM aligned with [1,1,1] (up to sign/scale)
    ok = False
    for e in res.efms_orig:
        nz = np.abs(e[np.abs(e) > 1e-12])
        if nz.size == 0:
            continue
        v = e / np.min(nz)
        if np.allclose(np.abs(v), np.array([1, 1, 1]), atol=1e-6):
            ok = True
            break
    assert ok

    # affinity bounds should be 0<=A<=Δμ for all three reactions (paper text)
    for b in res.affinity_bounds:
        assert b.lower <= 1e-9
        assert abs(b.upper - dm) < 1e-6
