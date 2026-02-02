"""Self-assembly CRN definition from arXiv:2407.11498 (Eq. self_assembly_reac).

Internal species:
  X1 (monomer), X2 (dimer), X3 (trimer)
External (chemostatted):
  F (fuel), W (waste)

Reactions:
  1) F + 2 X1 <-> X2 + W
  2) X1 + X2 <-> X3
  3) X3 <-> 3 X1

This file provides a MassActionCRN instance with stoichiometries.
"""

from __future__ import annotations

import numpy as np

from .model import MassActionCRN


def self_assembly_crn() -> MassActionCRN:
    # nX=3, nR=3
    # nu_plus = reactants
    nuX_plus = np.array([
        [2, 1, 0],  # X1
        [0, 1, 0],  # X2
        [0, 0, 1],  # X3
    ], dtype=float)

    nuX_minus = np.array([
        [0, 0, 3],  # X1
        [1, 0, 0],  # X2
        [0, 1, 0],  # X3
    ], dtype=float)

    # external species order: [F, W]
    # Reaction 1: F + 2X1 <-> X2 + W
    # So F is a reactant (plus), W is a product (minus).
    nuY_plus = np.array([
        [1, 0, 0],  # F
        [0, 0, 0],  # W
    ], dtype=float)

    nuY_minus = np.array([
        [0, 0, 0],  # F
        [1, 0, 0],  # W
    ], dtype=float)

    return MassActionCRN(
        nuX_plus=nuX_plus,
        nuX_minus=nuX_minus,
        nuY_plus=nuY_plus,
        nuY_minus=nuY_minus,
    )
