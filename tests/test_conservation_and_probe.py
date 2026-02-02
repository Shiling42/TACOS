"""Test conservation laws and probe generation on self-assembly network."""

from __future__ import annotations

import numpy as np

from crn_bounds.conservation import primitive_integer_basis
from crn_bounds.probes import generate_probe_candidates


def test_self_assembly_conservation_law():
    Sx = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=float)

    laws = primitive_integer_basis(Sx)
    # should contain r=[1,2,3]
    ok = any(np.allclose(r, np.array([1, 2, 3])) for r in laws)
    assert ok, laws


def test_self_assembly_probe_candidate():
    Sx = np.array([
        [-2, -1,  3],
        [ 1, -1,  0],
        [ 0,  1, -1],
    ], dtype=float)

    probes = generate_probe_candidates(Sx, coeff_bound=3)

    # should include [0,-3,2] up to scaling/sign
    target = np.array([0, -3, 2])
    def norm(v):
        v = v.astype(int)
        g = np.gcd.reduce(np.abs(v[v!=0])) if np.any(v!=0) else 1
        v = v // g
        for x in v:
            if x != 0:
                if x < 0:
                    v = -v
                break
        return v

    t = norm(target)
    ok = any(np.allclose(norm(p), t) for p in probes)
    assert ok, probes
