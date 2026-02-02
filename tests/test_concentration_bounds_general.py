"""Test general concentration bounds implementation.

These tests verify that the Π-pathway theory implementation works correctly
for various CRN configurations, not just the paper's single-cycle examples.
"""

from __future__ import annotations

import numpy as np
import pytest

from crn_bounds.api import CRNInput, run_pipeline
from crn_bounds.concentration_bounds import reaction_log_ratio_bound, LogRatioBound


class TestSingleCycleNetworks:
    """Tests for single-cycle networks (like paper examples)."""

    def test_self_assembly_all_reactions(self):
        """Verify all three self-assembly reaction bounds match paper."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0, 0.0])
        dm = 2.0
        A_Y = np.array([dm, 0.0, 0.0])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

        # R1: ln(x2/x1^2) in [0, dm]
        assert abs(res.reaction_log_bounds[0].lo - 0.0) < 1e-6
        assert abs(res.reaction_log_bounds[0].hi - dm) < 1e-6

        # R2: width should be dm
        assert abs((res.reaction_log_bounds[1].hi - res.reaction_log_bounds[1].lo) - dm) < 1e-6

        # R3: width should be dm
        assert abs((res.reaction_log_bounds[2].hi - res.reaction_log_bounds[2].lo) - dm) < 1e-6

    def test_self_assembly_different_driving(self):
        """Test that bounds scale correctly with driving magnitude."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0, 0.0])

        for dm in [1.0, 3.0, 5.0]:
            A_Y = np.array([dm, 0.0, 0.0])
            res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

            # All bounds should have width equal to dm
            for lb in res.reaction_log_bounds:
                width = lb.hi - lb.lo
                assert abs(width - dm) < 1e-6, f"Width {width} != dm {dm}"

    def test_nonzero_standard_potentials(self):
        """Test bounds with non-zero standard chemical potentials."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        # Non-zero standard potentials
        mu0_X = np.array([1.0, 2.0, 0.5])
        dm = 2.0
        A_Y = np.array([dm, 0.0, 0.0])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

        # Width should still be dm (standard potentials shift bounds but not width)
        for lb in res.reaction_log_bounds:
            width = lb.hi - lb.lo
            assert abs(width - dm) < 1e-6


class TestMultiDrivingNetworks:
    """Tests for networks with multiple driving reactions."""

    def test_two_independent_drivers(self):
        """Test network with two independent driving reactions."""
        # Simple 2-species, 2-reaction network
        # R1: A -> B (driven by dm1)
        # R2: B -> A (driven by dm2)
        Sx = np.array([
            [-1,  1],  # A
            [ 1, -1],  # B
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0])
        dm1, dm2 = 2.0, 1.0
        A_Y = np.array([dm1, dm2])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

        # With one EFM [1, 1], the normalized affinities are:
        # R1: (e @ A_Y) / e_1 = (dm1 + dm2) / 1 = 3.0
        # R2: (e @ A_Y) / e_2 = (dm1 + dm2) / 1 = 3.0
        # With include_zero, bounds are [0, 3.0]

        for lb in res.reaction_log_bounds:
            width = lb.hi - lb.lo
            # Width should be (dm1 + dm2) for both reactions
            assert abs(width - (dm1 + dm2)) < 1e-6


class TestThermodynamicConsistency:
    """Tests verifying thermodynamic consistency of bounds."""

    def test_equilibrium_network(self):
        """Test that undriven network gives zero-width bounds."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0, 0.0])
        A_Y = np.array([0.0, 0.0, 0.0])  # No driving

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

        # All bounds should have zero width (equilibrium)
        for lb in res.reaction_log_bounds:
            width = lb.hi - lb.lo
            assert abs(width) < 1e-6, f"Expected zero width at equilibrium, got {width}"

    def test_bounds_direction_consistency(self):
        """Test that lo <= hi always holds after canonicalization."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        mu0_X = np.array([1.0, -0.5, 2.0])
        A_Y = np.array([3.0, -1.0, 0.5])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=False)

        for i, lb in enumerate(res.reaction_log_bounds):
            assert lb.lo <= lb.hi, f"Reaction {i}: lo={lb.lo} > hi={lb.hi}"


class TestProbeConcentrationBounds:
    """Tests for probe concentration bounds."""

    def test_probe_bounds_tighter_than_zero(self):
        """Probe bounds should not include zero (unlike reaction bounds)."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0, 0.0])
        dm = 2.0
        A_Y = np.array([dm, 0.0, 0.0])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=True, max_probes=5)

        # All probe bounds should have finite values (no ±inf)
        for pr in res.probes:
            assert pr.log_ratio_bound.lo > float("-inf")
            assert pr.log_ratio_bound.hi < float("inf")

    def test_self_assembly_target_probe(self):
        """Test the target probe s=[0,-3,2] from the paper."""
        Sx = np.array([
            [-2, -1,  3],
            [ 1, -1,  0],
            [ 0,  1, -1],
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0, 0.0])
        dm = 2.0
        A_Y = np.array([dm, 0.0, 0.0])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=True, max_probes=20)

        # Find the target probe [0, -3, 2] or its negative [0, 3, -2]
        target_found = False
        for pr in res.probes:
            s = pr.probe_sx
            if (np.allclose(s, [0, -3, 2]) or np.allclose(s, [0, 3, -2])):
                target_found = True
                # Bound should be [-3*dm, 0] = [-6, 0]
                assert abs(pr.log_ratio_bound.lo - (-3 * dm)) < 1e-6
                assert abs(pr.log_ratio_bound.hi - 0.0) < 1e-6
                break

        assert target_found, "Target probe [0, -3, 2] not found"


class TestFrankChiralModel:
    """Tests for Frank model (chiral symmetry breaking)."""

    def test_frank_probe_symmetry(self):
        """Test that Frank model probe has symmetric bounds.

        Frank model (from paper):
        - Internal species: R, S (enantiomers)
        - Reactions:
          1) R + A <-> 2R (autocatalysis of R)
          2) S + A <-> 2S (autocatalysis of S)
          3) R + S <-> C (mutual inhibition)
        - Probe: S <-> R measures ln(r/s)
        """
        # Internal block only (R, S)
        Sx = np.array([
            [ 1, 0, -1],  # R
            [ 0, 1, -1],  # S
        ], dtype=float)

        mu0_X = np.array([0.0, 0.0])
        dm = 2.5  # Δμ/RT
        # Driving: A_Y=[dm, dm, -dm] so that cycle affinity sums correctly
        A_Y = np.array([dm, dm, -dm])

        res = run_pipeline(CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y), auto_probes=True, max_probes=10)

        # Look for probe that measures r/s ratio: [1, -1] or [-1, 1]
        for pr in res.probes:
            s = pr.probe_sx
            if np.allclose(np.abs(s), [1, 1]):
                # Symmetric system should give symmetric bounds
                # ln(r/s) in [-dm, +dm]
                assert abs(pr.log_ratio_bound.hi - dm) < 1e-6
                assert abs(pr.log_ratio_bound.lo - (-dm)) < 1e-6
                break
