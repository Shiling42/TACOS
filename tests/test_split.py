import numpy as np

from crn_bounds.stoichiometry import Stoichiometry
from crn_bounds.split import split_reversible


def test_split_shapes_and_mapping():
    # Two reactions, both reversible.
    Sx = np.array([
        [1.0, -2.0],
        [0.0,  1.0],
    ])
    sto = Stoichiometry(Sx=Sx)
    out = split_reversible(sto)

    assert out.stoich_split.Sx.shape == (2, 4)
    assert out.split_to_orig.tolist() == [0, 0, 1, 1]
    assert out.split_sign.tolist() == [1, -1, 1, -1]

    # Check columns are +/- original
    Sx_split = out.stoich_split.Sx
    assert np.allclose(Sx_split[:, 0], Sx[:, 0])
    assert np.allclose(Sx_split[:, 1], -Sx[:, 0])
    assert np.allclose(Sx_split[:, 2], Sx[:, 1])
    assert np.allclose(Sx_split[:, 3], -Sx[:, 1])
