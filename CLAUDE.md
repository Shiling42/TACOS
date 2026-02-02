# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python implementation of thermodynamic bounds machinery from arXiv:2407.11498 ("Thermodynamic space for chemical reaction networks"). Computes elementary flux modes (EFMs) and derives thermodynamic affinity/concentration bounds at steady state for chemical reaction networks.

## Build & Test Commands

```bash
# Install (from code/crn directory)
pip install -e .           # library only
pip install -e .[test]     # library + pytest

# Run tests
pytest -q                  # quick run
pytest tests/test_api_self_assembly_probes.py  # specific file

# Run scripts (examples)
python3 scripts/plot_self_assembly_from_efm.py --out output.png --DeltaMu_over_RT 2.0
python3 scripts/sample_self_assembly_relaxation_panels.py --outdir ./notes --n 600 --seed 0
```

## Architecture

### Core Pipeline Flow

`CRNInput` → `split_reversible()` → `enumerate_extreme_rays()` → `compute_affinity_bound()` → `reaction_log_ratio_bound()` → `PipelineResult`

### Key Modules (crn_bounds/)

- **api.py**: Main entry point. `CRNInput` dataclass for input, `run_pipeline()` orchestrates the full analysis, returns `PipelineResult`
- **split.py**: Converts reversible reactions to forward/backward irreversible representation for cddlib compatibility
- **efm.py**: EFM enumeration via `pycddlib` (polyhedral cone extreme rays)
- **affinity.py**: Steady-state affinity bounds from EFMs (paper Eq. 20/24)
- **concentration_bounds.py**: Log-ratio bounds `lo <= s^T ln x <= hi` for reactions and probes
- **conservation.py**: Exact rational nullspace (sympy) for conservation laws
- **probes.py**: Auto-detection of probe candidates from small-integer combinations
- **ldb.py**: Local detailed balance rate sampling for simulation validation
- **model.py**: Mass-action kinetics, ODE relaxation, steady-state root-finding

### Conventions

- All chemical potentials and affinities are **dimensionless** (normalized by RT): `mu0_X` = μ⁰/RT, `A_Y` = A^Y/RT
- Split-space uses irreversible representation internally; results mapped back to original coordinates
- Stoichiometry: `Sx` (internal species), `Sy` (external/chemostatted species)

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy ≥1.26 | Array operations |
| pycddlib ≥2.1.8 | Polyhedral extreme ray enumeration |
| sympy | Exact rational nullspace for conservation laws |
| scipy | Root-finding, ODE integration |
| pytest ≥8.0 | Testing |
| matplotlib | Scripts only (plotting) |

## Test Organization

- `test_api_*.py`: Integration tests using `run_pipeline()` on paper examples (self-assembly, Schlögl, Frank/chiral)
- `test_concentration_bounds_general.py`: Comprehensive tests for general Π-pathway concentration bounds
- `test_efm.py`, `test_split.py`, `test_affinity.py`: Unit tests for individual algorithms
- `test_conservation_and_probe.py`: Conservation law and probe generation tests

## Generated Figures

Run scripts to generate validation figures in `notes/`:
```bash
python3 scripts/plot_self_assembly_from_efm.py --out notes/fig_self_assembly_efm.png --DeltaMu_over_RT 2.0
python3 scripts/plot_schlogl_bounds.py --out notes/fig_schlogl_bounds.png
python3 scripts/plot_chiral_bounds.py --out notes/fig_chiral_bounds.png
python3 scripts/sample_self_assembly_relaxation_panels.py --outdir notes --n 300 --seed 42
```
