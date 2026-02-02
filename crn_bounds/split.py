from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .stoichiometry import Stoichiometry


@dataclass(frozen=True)
class SplitResult:
    """Reversible-split mapping.

    For each original reaction rho we create up to two split reactions:
      rho+  with column +S[:,rho]
      rho-  with column -S[:,rho]

    The bookkeeping arrays map split indices -> original index and sign.
    """

    stoich_split: Stoichiometry
    split_to_orig: np.ndarray  # (nR_split,) int
    split_sign: np.ndarray     # (nR_split,) in {+1,-1}


def split_reversible(stoich: Stoichiometry, reversible: np.ndarray | None = None) -> SplitResult:
    """Split reversible reactions into an all-irreversible representation.

    Parameters
    - stoich: original Sx/Sy
    - reversible: boolean mask (nR,) indicating which reactions are reversible.
      If None: assume all are reversible.

    Returns
    - SplitResult with Sx_split (and Sy_split if present).

    Notes
    - This is purely a *representation change* for EFM/extreme-ray enumeration.
    - Net flux mapping back: v_orig[rho] = sum_{k:orig=rho} split_sign[k] * v_split[k]
    """

    Sx = stoich.Sx
    Sy = stoich.Sy
    nR = stoich.nR

    if reversible is None:
        reversible = np.ones(nR, dtype=bool)
    reversible = np.asarray(reversible, dtype=bool)
    if reversible.shape != (nR,):
        raise ValueError("reversible mask must have shape (nR,)")

    cols_x = []
    cols_y = []
    split_to_orig = []
    split_sign = []

    for rho in range(nR):
        # Always include forward (rho+)
        cols_x.append(Sx[:, rho])
        if Sy is not None:
            cols_y.append(Sy[:, rho])
        split_to_orig.append(rho)
        split_sign.append(+1)

        # Include backward only if reversible
        if reversible[rho]:
            cols_x.append(-Sx[:, rho])
            if Sy is not None:
                cols_y.append(-Sy[:, rho])
            split_to_orig.append(rho)
            split_sign.append(-1)

    Sx_split = np.stack(cols_x, axis=1)
    Sy_split = None if Sy is None else np.stack(cols_y, axis=1)

    return SplitResult(
        stoich_split=Stoichiometry(Sx=Sx_split, Sy=Sy_split),
        split_to_orig=np.asarray(split_to_orig, dtype=int),
        split_sign=np.asarray(split_sign, dtype=int),
    )
