"""Conservation laws (left nullspace) utilities.

Goal (User0 request):
A) Extract a *primitive / integer* basis for conservation laws.
B) Use these laws to generate canonical probe directions in reaction space.

Given Sx (nX x nR), conservation laws are r such that r^T Sx = 0.

We provide:
- rational_nullspace_left(): rational basis via sympy
- primitive_integer_basis(): scale vectors to integer primitive form

NOTE: This is intended for small/moderate networks (paper examples).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from functools import reduce

import numpy as np
from numpy.typing import NDArray


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else abs(a or b)


def _lcm_list(xs: list[int]) -> int:
    return reduce(_lcm, xs, 1)


def _gcd_list(xs: list[int]) -> int:
    xs = [abs(x) for x in xs if x != 0]
    return reduce(gcd, xs, 0) if xs else 1


def _primitive_int(vec: NDArray[np.int64]) -> NDArray[np.int64]:
    g = _gcd_list(vec.tolist())
    v = vec // g
    # make first nonzero positive
    for x in v:
        if x != 0:
            if x < 0:
                v = -v
            break
    return v


def rational_nullspace_left(Sx: NDArray[np.float64]) -> list[list[Fraction]]:
    """Compute a rational basis for left nullspace of Sx.

    Returns a list of vectors r (as Fractions) such that r^T Sx = 0.

    Uses sympy for exact nullspace on integers/rationals.
    """
    import sympy as sp

    S = sp.Matrix(Sx)
    # left nullspace of Sx: nullspace of Sx^T
    ns = (S.T).nullspace()
    out: list[list[Fraction]] = []
    for v in ns:
        fr = []
        for x in list(v):
            # sympy Rational -> Fraction; otherwise coerce via str
            if isinstance(x, sp.Rational):
                fr.append(Fraction(int(x.p), int(x.q)))
            else:
                fr.append(Fraction(str(x)))
        out.append(fr)
    return out


def primitive_integer_basis(Sx: NDArray[np.float64], *, max_den: int = 1_000_000) -> list[NDArray[np.int64]]:
    """Return a primitive integer basis for conservation laws.

    We compute an exact rational nullspace (sympy). Then for each vector:
    - convert each entry to a Fraction with bounded denominator
    - scale by lcm of denominators
    - divide by gcd to get a primitive integer vector

    Args:
      max_den: bound denominators when converting to Fraction

    Returns:
      list of primitive integer vectors r (nX,)
    """
    basis_q = rational_nullspace_left(Sx)
    basis_i: list[NDArray[np.int64]] = []
    for v in basis_q:
        v2 = [f.limit_denominator(max_den) for f in v]
        dens = [f.denominator for f in v2]
        L = _lcm_list(dens)
        ints = np.array([int(f * L) for f in v2], dtype=np.int64)
        basis_i.append(_primitive_int(ints))
    return basis_i
