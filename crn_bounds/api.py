"""Public API for the paper-style bounds pipeline.

Standardized input (as requested by User0):
- Stoichiometric matrix split into internal/external blocks: (Sx, Sy)
- Internal standard chemical potentials: mu0_X (dimensionless μ/RT)
- External driving per reaction: A_Y (dimensionless A^Y/RT)

Optional:
- reversible mask
- chemical probes (additional reaction columns) for reaction-space bounds

This module defines:
- CRNInput dataclass
- PipelineResult dataclass
- run_pipeline() entrypoint (bounds-only; no kinetics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .stoichiometry import Stoichiometry
from .split import split_reversible
from .efm import enumerate_extreme_rays
from .affinity import compute_affinity_bound
from .probes import generate_probe_candidates
from .concentration_bounds import probe_log_ratio_bound, reaction_log_ratio_bound, LogRatioBound
from .utils import map_split_efms_to_original


@dataclass(frozen=True)
class CRNInput:
    Sx: NDArray[np.float64]                 # (nX, nR)
    mu0_X: NDArray[np.float64]              # (nX,)
    A_Y: NDArray[np.float64]                # (nR,)  external affinity per reaction (A^Y/RT)
    Sy: NDArray[np.float64] | None = None   # (nY, nR) optional
    reversible: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class ReactionAffinityBounds:
    lower: float
    upper: float


@dataclass(frozen=True)
class ProbeResult:
    probe_sx: NDArray[np.int64]               # (nX,) internal stoich of probe
    efms_orig: list[NDArray[np.float64]]      # EFMs of extended network (mapped to original+probe coords)
    affinity_bound: ReactionAffinityBounds    # bound for the probe reaction
    log_ratio_bound: LogRatioBound            # bound on s^T ln x for this probe


@dataclass(frozen=True)
class PipelineResult:
    efms_split: list[NDArray[np.float64]]
    efms_orig: list[NDArray[np.float64]]
    affinity_bounds: list[ReactionAffinityBounds]
    reaction_log_bounds: list[LogRatioBound]
    probes: list[ProbeResult]


def run_pipeline(
    inp: CRNInput,
    *,
    auto_probes: bool = False,
    probe_coeff_bound: int = 3,
    max_probes: int = 10,
) -> PipelineResult:
    """Run the bounds-only pipeline:

    1) split reversible
    2) enumerate EFMs/extreme rays on split cone (Sx_split v=0, v>=0)
    3) map EFMs back to original reaction coordinates
    4) compute affinity bounds Eq.(20) for each reaction using A_Y

    Returns:
      PipelineResult
    """
    Sx = np.asarray(inp.Sx, dtype=float)
    mu0_X = np.asarray(inp.mu0_X, dtype=float)
    A_Y = np.asarray(inp.A_Y, dtype=float)

    nX, nR = Sx.shape
    if mu0_X.shape != (nX,):
        raise ValueError("mu0_X must have shape (nX,)")
    if A_Y.shape != (nR,):
        raise ValueError("A_Y must have shape (nR,)")

    sto = Stoichiometry(Sx=Sx, Sy=inp.Sy)
    split = split_reversible(sto, reversible=inp.reversible)

    efms_split = enumerate_extreme_rays(split.stoich_split.Sx)

    # map back to original coordinates (filter zeros; keep a canonical sign)
    efms_orig = map_split_efms_to_original(
        efms_split, split.split_to_orig, split.split_sign, nR
    )

    # compute bounds for each rho
    bounds: list[ReactionAffinityBounds] = []
    for rho in range(nR):
        lo, hi = compute_affinity_bound(rho, A=A_Y, E=efms_orig, include_zero=True)
        bounds.append(ReactionAffinityBounds(lower=lo, upper=hi))

    # reaction concentration (log-ratio) bounds
    reaction_log_bounds: list[LogRatioBound] = []
    for rho in range(nR):
        reaction_log_bounds.append(
            reaction_log_ratio_bound(
                rho=rho,
                Sx=Sx,
                mu0_X=mu0_X,
                A_Y=A_Y,
                efms=efms_orig,
            )
        )

    probes: list[ProbeResult] = []
    if auto_probes:
        cand = generate_probe_candidates(Sx, coeff_bound=probe_coeff_bound, max_candidates=max_probes)

        for p in cand:
            # extend network with probe column
            Sx_ext = np.hstack([Sx, p.reshape(-1, 1).astype(float)])
            A_Y_ext = np.concatenate([A_Y, [0.0]])  # probe has no external driving

            sto_ext = Stoichiometry(Sx=Sx_ext, Sy=None)
            split_ext = split_reversible(sto_ext, reversible=None)  # assume reversible
            efms_split_ext = enumerate_extreme_rays(split_ext.stoich_split.Sx)

            # map back to original+probe coordinates
            nR_ext = nR + 1
            efms_orig_ext = map_split_efms_to_original(
                efms_split_ext, split_ext.split_to_orig, split_ext.split_sign, nR_ext,
                canonicalize=False, deduplicate=False  # Preserve original probe behavior
            )

            # bound for probe reaction is index nR
            lo, hi = compute_affinity_bound(nR, A=A_Y_ext, E=efms_orig_ext, include_zero=False)

            # concentration log-ratio bound for probe
            lr = probe_log_ratio_bound(
                s_probe=p.astype(float),
                mu0_X=mu0_X,
                A_Y_ext=A_Y_ext,
                efms_ext=efms_orig_ext,
                probe_index=nR,
            )

            probes.append(
                ProbeResult(
                    probe_sx=p,
                    efms_orig=efms_orig_ext,
                    affinity_bound=ReactionAffinityBounds(lower=lo, upper=hi),
                    log_ratio_bound=lr,
                )
            )

    return PipelineResult(
        efms_split=efms_split,
        efms_orig=efms_orig,
        affinity_bounds=bounds,
        reaction_log_bounds=reaction_log_bounds,
        probes=probes,
    )
