"""CRN thermodynamic bounds (arXiv:2407.11498) — skeleton.

Core contract:
- inputs: stoichiometry split into internal X / external Y blocks
- workflow: reversible split -> flux cone -> EFMs -> thermodynamic bounds

The equation mapping lives in: papers/arxiv-2407.11498/notes/DERIVATION.md
"""

from .stoichiometry import Stoichiometry
from .split import split_reversible
