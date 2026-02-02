"""EFM / extreme-ray enumeration via cddlib (pycddlib).

We want extreme rays of the flux cone:
    Sx_split v = 0,  v >= 0

This is a polyhedral cone in R^{nR}.

cddlib works with H-representation (inequalities/equalities):
    A x >= 0
We encode:
  - v_i >= 0  for all i
  - Sx_split v = 0  as two inequalities:  Sx_split v >= 0 and -Sx_split v >= 0

Then compute generators (V-representation) and return rays.

Notes:
- cddlib uses exact rational arithmetic if we provide fractions.
- We'll accept float input but convert to rationals with a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np


def _to_fraction_matrix(A: np.ndarray, max_den: int = 10_000) -> list[list[Fraction]]:
    out: list[list[Fraction]] = []
    for row in A:
        out.append([Fraction(x).limit_denominator(max_den) for x in row])
    return out


def enumerate_extreme_rays(
    Sx_split: np.ndarray,
    *,
    max_den: int = 10_000,
    tol: float = 1e-12,
    normalize: bool = True,
) -> list[np.ndarray]:
    """Enumerate extreme rays (EFMs) for cone {v | Sx v = 0, v>=0}.

    Args:
        Sx_split: (nX, nR)
        max_den: max denominator for rational conversion
        tol: numerical tolerance for filtering tiny negatives
        normalize: if True, scale each ray by L1 norm

    Returns:
        list of rays, each (nR,) float64 with v>=0.
    """

    try:
        import cdd  # pycddlib
    except Exception as e:
        raise ImportError(
            "pycddlib is required for enumerate_extreme_rays. Install pycddlib and libcdd-dev."
        ) from e

    S = np.asarray(Sx_split, dtype=float)
    nX, nR = S.shape

    # Build inequality matrix for A x >= 0.
    # cdd expects a matrix with leading column = b, so format is [b | A].
    rows: list[list[Fraction]] = []

    # v_i >= 0
    for i in range(nR):
        a = np.zeros(nR)
        a[i] = 1.0
        rows.append([Fraction(0)] + [Fraction(x).limit_denominator(max_den) for x in a])

    # S v >= 0 and -S v >= 0
    for sign in (+1.0, -1.0):
        for r in range(nX):
            a = sign * S[r, :]
            rows.append([Fraction(0)] + [Fraction(x).limit_denominator(max_den) for x in a])

    # cdd: build H-representation with real numbers (good enough for our integer S).
    A = np.array([[float(x) for x in row] for row in rows], dtype=float)
    mat = cdd.matrix_from_array(A, rep_type=cdd.RepType.INEQUALITY)

    poly = cdd.polyhedron_from_matrix(mat)
    gen = cdd.copy_generators(poly)  # V-representation as Matrix

    # gen.array rows: [b | x...]
    G = np.array(gen.array, dtype=float)

    rays: list[np.ndarray] = []
    for row in G:
        b = row[0]
        vec = row[1:].copy()

        if abs(b) > tol:
            continue

        vec[np.abs(vec) < tol] = 0.0
        if np.any(vec < -tol):
            continue
        vec = np.maximum(vec, 0.0)

        if normalize:
            s = vec.sum()
            if s > tol:
                vec = vec / s

        rays.append(vec)

    # Deduplicate (within tol) by rounding
    uniq = {}
    for v in rays:
        key = tuple(np.round(v, 12))
        uniq[key] = v

    return list(uniq.values())