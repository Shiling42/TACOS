"""Test EFM/extreme ray enumeration using pycddlib.

We test a simple cone:
  S v = 0, v>=0
with S = [1, -1], so v1=v2>=0.
Extreme ray should be proportional to [1,1].
"""

from __future__ import annotations

import numpy as np

from crn_bounds.efm import enumerate_extreme_rays


def test_simple_equal_fluxes():
    S = np.array([[1.0, -1.0]])
    rays = enumerate_extreme_rays(S)
    assert len(rays) >= 1

    # Find a ray close to [0.5, 0.5] after L1 normalization
    target = np.array([0.5, 0.5])
    ok = any(np.allclose(r, target, atol=1e-8) for r in rays)
    assert ok, rays
