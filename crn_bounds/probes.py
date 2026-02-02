"""Generate canonical probe directions from conservation laws.

We want probe stoichiometries s_hat (in internal species space) such that:
  r^T s_hat = 0 for all conservation laws r
and s_hat is not in the span of existing reactions as a column already.

For now, we generate *small integer* candidates in the reaction space
(colspace of Sx) by searching integer combinations of reactions with
small coefficients, and filter those that are not equal to any existing
reaction column (up to scaling).

This matches paper's self-assembly example where the probe is:
  s_hat = [0,-3,2]^T
which equals an integer combination of the original reactions:
  s_hat = 1*S1 - 3*S2 - 1*S3  (check)

This is a pragmatic algorithm for small networks.
"""

from __future__ import annotations

import itertools
import numpy as np
from numpy.typing import NDArray

from .conservation import primitive_integer_basis
from .utils import canonicalize_vector


def _primitive_col(v: NDArray[np.int64]) -> NDArray[np.int64]:
    """Make an integer vector primitive (divide by GCD, first nonzero positive)."""
    result = canonicalize_vector(v.astype(float), make_primitive=True)
    return result.astype(np.int64)


def generate_probe_candidates(
    Sx: NDArray[np.float64],
    *,
    coeff_bound: int = 3,
    max_candidates: int = 50,
) -> list[NDArray[np.int64]]:
    """Generate canonical probe stoichiometry vectors in internal space.

    Args:
      Sx: (nX,nR)
      coeff_bound: search coefficients c_r in [-coeff_bound, ..., coeff_bound]
      max_candidates: cap returned list

    Returns:
      list of primitive integer vectors s_hat (nX,)
    """
    Sx = np.asarray(Sx, dtype=float)
    nX, nR = Sx.shape

    # existing reaction columns in primitive integer form (Sx is integer in paper examples)
    cols = []
    for rho in range(nR):
        ints = np.asarray(Sx[:, rho], dtype=np.int64)
        cols.append(_primitive_col(ints))

    colset = {tuple(c.tolist()) for c in cols}

    candidates: list[NDArray[np.int64]] = []

    # brute force small integer combinations of columns
    rng = range(-coeff_bound, coeff_bound + 1)
    for coeffs in itertools.product(rng, repeat=nR):
        if all(c == 0 for c in coeffs):
            continue
        c = np.array(coeffs, dtype=np.int64)
        v = (Sx.astype(np.int64) @ c)  # exact integer combination
        if not np.any(v):
            continue
        p = _primitive_col(v.astype(np.int64))
        t = tuple(p.tolist())
        if t in colset:
            continue
        if any(np.array_equal(p, q) for q in candidates):
            continue
        candidates.append(p)
        if len(candidates) >= max_candidates:
            break

    # ensure orthogonal to conservation laws
    laws = primitive_integer_basis(Sx)
    out = []
    for p in candidates:
        if all(int(r @ p) == 0 for r in laws):
            out.append(p)

    return out