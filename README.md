# TACOS: Thermodynamic Bounds for Chemical Reaction Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2407.11498-b31b1b.svg)](https://arxiv.org/abs/2407.11498)

**TACOS** (Thermodynamic space for Accessible Concentrations of Out-of-equilibrium Stationary chemical reaction networks) is a Python package for computing fundamental thermodynamic bounds on chemical reaction networks (CRNs) at steady state.

This repository provides the official implementation accompanying the paper:

> **Thermodynamic Space of Chemical Reaction Networks**
> arXiv:2407.11498

## Overview

Chemical reaction networks driven out of equilibrium by external reservoirs (chemostats) can exhibit rich behaviors including bistability, oscillations, and symmetry breaking. However, thermodynamics imposes fundamental constraints on these behaviors.

TACOS computes these constraints by:
1. **Elementary Flux Mode (EFM) enumeration** via polyhedral geometry
2. **Affinity bounds** from cycle affinities (Paper Eq. 20)
3. **Concentration bounds** via Π-pathway theory
4. **Chemical probe analysis** for virtual species interconversions

### Key Results

For any CRN at steady state:
- **Reaction affinities** are bounded by the extremal cycle affinities of EFMs passing through that reaction
- **Concentration ratios** are constrained by the thermodynamic driving and network topology
- **Chemical probes** (virtual interconversions) have bounded log-ratios even when individual concentrations are unconstrained

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tacos.git
cd tacos

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install with all dependencies
pip install -e ".[full]"
```

### Installing pycddlib

The geometric EFM enumeration requires `pycddlib`, which depends on the cddlib C library.

**macOS (Homebrew):**
```bash
brew install cddlib gmp
CFLAGS="-I$(brew --prefix cddlib)/include -I$(brew --prefix gmp)/include" \
LDFLAGS="-L$(brew --prefix cddlib)/lib -L$(brew --prefix gmp)/lib" \
pip install pycddlib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libcdd-dev libgmp-dev
pip install pycddlib
```

### Minimal Install

If you don't need geometric EFM enumeration:
```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from crn_bounds.api import CRNInput, run_pipeline

# Define a fuel-driven self-assembly network
# R1: F + 2A ⇌ B + W (fuel-driven dimerization)
# R2: A + B ⇌ C      (trimerization)
# R3: C ⇌ 3A         (disassembly)
Sx = np.array([
    [-2, -1,  3],  # A: consumed in R1,R2; produced in R3
    [ 1, -1,  0],  # B: produced in R1; consumed in R2
    [ 0,  1, -1],  # C: produced in R2; consumed in R3
], dtype=float)

# Standard chemical potentials (dimensionless: μ⁰/RT)
mu0_X = np.array([0.0, 0.0, 0.0])

# External driving: Δμ/RT = 2.0 on first reaction (fuel hydrolysis)
A_Y = np.array([2.0, 0.0, 0.0])

# Run the analysis pipeline
result = run_pipeline(
    CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y),
    auto_probes=True
)

# Print affinity bounds for each reaction
print("Affinity Bounds (A/RT):")
for i, ab in enumerate(result.affinity_bounds):
    print(f"  R{i+1}: [{ab.lower:.3f}, {ab.upper:.3f}]")

# Print concentration log-ratio bounds
print("\nConcentration Bounds (s·ln x):")
for i, lb in enumerate(result.reaction_log_bounds):
    print(f"  R{i+1}: [{lb.lo:.3f}, {lb.hi:.3f}]")

# Print probe results (if auto_probes=True)
print("\nProbe Bounds:")
for pr in result.probes:
    print(f"  s={pr.probe_sx}: ln-ratio ∈ [{pr.log_ratio_bound.lo:.3f}, {pr.log_ratio_bound.hi:.3f}]")
```

## Demonstrations

The `demos/` directory contains four complete examples:

| Demo | Network | Key Concept |
|------|---------|-------------|
| **demo_01** | Fuel-driven self-assembly | Affinity & concentration bounds with simulation validation |
| **demo_02** | Schlögl model | Bistability (saddle-node bifurcation) within thermodynamic bounds |
| **demo_03** | Frank model (chiral) | Pitchfork bifurcation and symmetry breaking limits |
| **demo_04** | EFM methods | Comparison of combinatorial vs geometric algorithms |

### Running Demos

```bash
# Run all demos
cd code/crn
python demos/demo_01_self_assembly.py
python demos/demo_02_schlogl.py
python demos/demo_03_chiral.py
python demos/demo_04_efm_methods.py

# Generate SVG output (optional)
python demos/demo_01_self_assembly.py --svg
```

Output figures are saved to `notes/`.

See [`demos/README.md`](demos/README.md) for detailed documentation.

## Paper Examples

### 1. Self-Assembly (Section IV.A)

```bash
# Concentration bound panels with simulation validation
python scripts/sample_self_assembly_relaxation_panels.py \
    --outdir notes --DeltaMu_over_RT 2.0 --n 300 --seed 42

# Affinity bound visualization
python scripts/plot_affinity_bounds_validation.py \
    --outdir notes --DeltaMu_over_RT 2.0 --n 300
```

### 2. Schlögl Model (Section IV.B)

```bash
python scripts/plot_schlogl_bounds.py --out notes/fig_schlogl_bounds.png
```

### 3. Chiral Symmetry Breaking (Section IV.C)

```bash
python scripts/plot_chiral_bounds.py --out notes/fig_chiral_bounds.png
```

### 4. EFM Enumeration Complexity (Appendix A)

```bash
python scripts/benchmark_efm_enumeration.py --out notes/fig_EFM_search.png --n-max 12
```

## Repository Structure

```
tacos/
├── crn_bounds/                  # Core library
│   ├── api.py                   # Main API: CRNInput, run_pipeline
│   ├── efm.py                   # EFM enumeration via pycddlib
│   ├── affinity.py              # Affinity bounds computation
│   ├── concentration_bounds.py  # Log-ratio bounds (Π-pathway theory)
│   ├── probes.py                # Auto-probe detection
│   ├── conservation.py          # Conservation law computation
│   ├── split.py                 # Reversible reaction splitting
│   ├── ldb.py                   # Local detailed balance sampling
│   ├── model.py                 # Mass-action kinetics & ODE relaxation
│   └── self_assembly.py         # Self-assembly CRN definition
│
├── demos/                       # Demonstration scripts
│   ├── README.md                # Demo documentation
│   ├── demo_01_self_assembly.py # Fuel-driven self-assembly
│   ├── demo_02_schlogl.py       # Schlögl bistability
│   ├── demo_03_chiral.py        # Chiral symmetry breaking
│   └── demo_04_efm_methods.py   # EFM algorithm comparison
│
├── scripts/                     # Analysis & plotting scripts
├── tests/                       # Test suite (22 tests)
├── docs/                        # Algorithm documentation
│   └── EFM_ALGORITHMS.md        # Detailed EFM algorithm descriptions
│
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## API Reference

### CRNInput

The input specification for a chemical reaction network:

```python
@dataclass
class CRNInput:
    Sx: np.ndarray           # Internal stoichiometry matrix (n_species × n_reactions)
    mu0_X: np.ndarray        # Standard chemical potentials (n_species,), dimensionless μ⁰/RT
    A_Y: np.ndarray          # External affinity per reaction (n_reactions,), dimensionless A^Y/RT
    Sy: np.ndarray | None    # External stoichiometry (optional)
    reversible: np.ndarray | None  # Reversibility mask (optional, default: all reversible)
```

### run_pipeline

The main analysis function:

```python
def run_pipeline(
    inp: CRNInput,
    auto_probes: bool = False,      # Auto-detect chemical probes
    probe_coeff_bound: int = 3,     # Max coefficient for probe candidates
    max_probes: int = 10,           # Max number of probes to analyze
) -> PipelineResult
```

### PipelineResult

The output containing all computed bounds:

```python
@dataclass
class PipelineResult:
    efms_split: list[np.ndarray]           # EFMs in split (forward/backward) representation
    efms_orig: list[np.ndarray]            # EFMs in original coordinates
    affinity_bounds: list[ReactionAffinityBounds]  # Affinity bounds per reaction
    reaction_log_bounds: list[LogRatioBound]       # Concentration bounds per reaction
    probes: list[ProbeResult]              # Results for chemical probes
```

### Supporting Classes

```python
@dataclass
class ReactionAffinityBounds:
    lower: float    # Lower bound on A_ρ/RT
    upper: float    # Upper bound on A_ρ/RT

@dataclass
class LogRatioBound:
    s: np.ndarray   # Stoichiometric vector
    lo: float       # Lower bound on s·ln(x)
    hi: float       # Upper bound on s·ln(x)

@dataclass
class ProbeResult:
    probe_sx: np.ndarray         # Probe stoichiometry
    efms_orig: list[np.ndarray]  # EFMs containing this probe
    affinity_bound: ReactionAffinityBounds
    log_ratio_bound: LogRatioBound
```

## Algorithm Details

### Elementary Flux Modes (EFMs)

EFMs are the minimal steady-state flux patterns in a reaction network. We support two enumeration methods:

1. **Geometric method** (default): Polyhedral extreme-ray enumeration via `pycddlib`
   - Output-sensitive complexity
   - Recommended for networks with n > 6 reactions

2. **Combinatorial method**: Circuit enumeration via SVD nullspace
   - O(C(2n, r)) complexity
   - Only practical for small networks (n ≤ 6)

See [`docs/EFM_ALGORITHMS.md`](docs/EFM_ALGORITHMS.md) for detailed descriptions.

### Affinity Bounds (Paper Eq. 20)

For a reaction ρ, the steady-state affinity is bounded by:

```
min_e (e·A_Y / e_ρ) ≤ A_ρ^ss ≤ max_e (e·A_Y / e_ρ)
```

where the min/max are over all EFMs e with e_ρ ≠ 0.

### Concentration Bounds (Π-pathway Theory)

For a reaction with internal stoichiometry s:

```
s·ln(x^ss) ∈ [-ΔG⁰/RT - (A_max - A_Y), -ΔG⁰/RT - (A_min - A_Y)]
```

where ΔG⁰ = s·μ⁰ is the standard free energy change.

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_api_self_assembly.py -v
```

All 22 tests cover:
- EFM enumeration correctness
- Affinity bound computation
- Concentration bounds (general Π-pathway theory)
- Paper examples (self-assembly, Schlögl, Frank model)

## Conventions

### Dimensionless Units

All quantities are normalized by RT:
- `mu0_X`: μ⁰/RT (standard chemical potentials)
- `A_Y`: A^Y/RT (external affinities)
- Affinity bounds: A/RT
- Log-ratio bounds: dimensionless

### Sign Conventions

- **Stoichiometry**: negative for reactants, positive for products
- **Affinity**: A = -ΔG/RT > 0 for spontaneous reactions
- **EFMs**: non-negative in split representation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{thermodynamic_space_2024,
  title={Thermodynamic Space of Chemical Reaction Networks},
  author={...},
  journal={arXiv preprint arXiv:2407.11498},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the Swiss National Science Foundation and STARS@UNIPD.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

For major changes, please open an issue first to discuss what you would like to change.
