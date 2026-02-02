from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Stoichiometry:
    """Paper-aligned stoichiometry container.

    We follow the paper notation:
      S = (S^X; S^Y)

    Shapes:
      Sx: (nX, nR)
      Sy: (nY, nR) or None

    All reactions here are in the *original* (possibly reversible) representation.
    Use split_reversible() to build the split irreversible representation.
    """

    Sx: np.ndarray
    Sy: np.ndarray | None = None

    def __post_init__(self):
        Sx = np.asarray(self.Sx, dtype=float)
        object.__setattr__(self, "Sx", Sx)
        if self.Sy is not None:
            Sy = np.asarray(self.Sy, dtype=float)
            if Sy.shape[1] != Sx.shape[1]:
                raise ValueError(f"Sy has nR={Sy.shape[1]} but Sx has nR={Sx.shape[1]}")
            object.__setattr__(self, "Sy", Sy)

    @property
    def nX(self) -> int:
        return int(self.Sx.shape[0])

    @property
    def nR(self) -> int:
        return int(self.Sx.shape[1])

    @property
    def nY(self) -> int:
        return 0 if self.Sy is None else int(self.Sy.shape[0])
