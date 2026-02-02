# TACOS Demonstrations

This directory contains four demonstration scripts that showcase the key capabilities of the TACOS package for computing thermodynamic bounds on chemical reaction networks.

All demos are based on examples from arXiv:2407.11498.

## Quick Start

```bash
cd code/crn
pip install -e ".[full]"

# Run all demos
python demos/demo_01_self_assembly.py
python demos/demo_02_schlogl.py
python demos/demo_03_chiral.py
python demos/demo_04_efm_methods.py
```

Output figures are saved to `notes/` directory.

---

## Demo 1: Fuel-Driven Self-Assembly

**File:** `demo_01_self_assembly.py`

The primary example from the paper demonstrating thermodynamic bounds on a dissipative self-assembly network.

### Network

```
R1: F + 2A ⇌ B + W   (fuel-driven dimerization)
R2: A + B ⇌ C        (trimerization)
R3: C ⇌ 3A           (disassembly)
```

Where F (fuel) and W (waste) are external chemostats providing thermodynamic driving.

### What This Demo Shows

1. **Affinity bounds**: Upper and lower limits on reaction affinities at steady state
2. **Concentration bounds**: Constraints on log-concentration ratios
3. **Simulation validation**: 800 LDB-consistent steady states shown as scatter points

### Key Concept

The scatter points (simulated steady states) all fall **within** the computed thermodynamic bounds, validating that these bounds are universal constraints independent of kinetic details.

### Output

- `notes/demo_self_assembly.png` (default)
- `notes/demo_self_assembly.svg` (with `--svg` flag)

### Run

```bash
python demos/demo_01_self_assembly.py
python demos/demo_01_self_assembly.py --svg  # Also generate SVG
```

---

## Demo 2: Schlögl Model (Bistability)

**File:** `demo_02_schlogl.py`

A classic autocatalytic network exhibiting bistability via saddle-node bifurcation.

### Network

```
R1: 2X + A ⇌ 3X   (autocatalytic production)
R2: X ⇌ B         (degradation)
```

Where A and B are external chemostats with concentrations a and b.

### What This Demo Shows

1. **Bifurcation diagram**: Steady-state concentration ln(x) vs cycle affinity A_e/RT = ln(a/b)
2. **Thermodynamic bounds**: The allowed region ln(x) ∈ [ln(b), ln(b) + A_e/RT]
3. **Saddle-node bifurcation**: At A_e/RT ≈ 5.7, bistability emerges with two stable branches

### Key Concept

Both stable branches of the bifurcation diagram remain **within** the thermodynamic bounds. The lower branch approaches the lower bound ln(x) = ln(b), while the upper branch approaches ln(x) = ln(a).

### Parameters

- k₁⁺ = k₁⁻ = 1 (autocatalysis)
- k₂⁺ = k₂⁻ = 8 (degradation)
- b = 0.02 (fixed waste concentration)

### Output

- `notes/demo_schlogl.png` (default)
- `notes/demo_schlogl.svg` (with `--svg` flag)

### Run

```bash
python demos/demo_02_schlogl.py
python demos/demo_02_schlogl.py --svg
```

---

## Demo 3: Chiral Symmetry Breaking (Frank Model)

**File:** `demo_03_chiral.py`

Demonstrates thermodynamic limits on enantiomeric excess in a chiral autocatalysis network.

### Network

```
R1: R + A ⇌ 2R   (R-enantiomer autocatalysis)
R2: S + A ⇌ 2S   (S-enantiomer autocatalysis)
R3: R + S ⇌ C    (mutual inhibition / cross-catalysis)
```

This is the Frank model for homochirality emergence.

### What This Demo Shows

1. **Pitchfork bifurcation**: Racemic state (ln(r/s) = 0) becomes unstable, leading to chiral states
2. **Thermodynamic bounds**: ln(r/s) ∈ [-Δμ/RT, +Δμ/RT]
3. **Standard potential contribution**: The driving Δμ/RT includes both concentration and rate constant contributions

### Key Concept

Perfect homochirality (|ln(r/s)| → ∞) would require infinite thermodynamic driving. The chiral branches asymptotically approach but never exceed the bounds ±Δμ/RT.

### Important Note

The thermodynamic driving includes standard potential contributions from rate constant ratios:
```
Δμ/RT = 2 ln(k₀⁺/k₀⁻) + ln(k₁⁺/k₁⁻) + ln(a²/c)
```

Not just the naive ln(a²/c).

### Parameters

- k₀⁺ = 1.0, k₀⁻ = 0.7 (autocatalysis)
- k₁⁺ = 16.0, k₁⁻ = 2.0 (mutual inhibition)
- c = 0.5 (product C concentration)

### Output

- `notes/demo_chiral.png` (default)
- `notes/demo_chiral.svg` (with `--svg` flag)

### Run

```bash
python demos/demo_03_chiral.py
python demos/demo_03_chiral.py --svg
```

---

## Demo 4: EFM Enumeration Methods

**File:** `demo_04_efm_methods.py`

Compares two algorithms for Elementary Flux Mode (EFM) enumeration.

### Test Network

Modular self-assembly network with n species:
```
A → B₁         (activation)
B₁ + Bₘ → Bₘ₊₁  (assembly, m = 1,...,n-1)
Bₘ → m·A       (degradation, m = 1,...,n)
```

### Algorithms Compared

| Method | Complexity | Practical Range |
|--------|------------|-----------------|
| **Combinatorial** | O(C(2n, r)) | n ≤ 6 |
| **Geometric** | Output-sensitive | n > 15 |

1. **Combinatorial method**: Circuit enumeration via SVD nullspace
   - Enumerates all column subsets of split stoichiometry matrix
   - Checks for 1D nullspace with conformal (sign-consistent) support

2. **Geometric method**: Polyhedral extreme ray enumeration via pycddlib
   - Formulates flux cone as H-representation
   - Uses Double Description algorithm for V-representation

### What This Demo Shows

1. **Panel (a)**: Network schematic
2. **Panel (b)**: Computation time vs network size (log scale)
3. **Panel (c)**: Number of EFMs vs network size (log scale)

### Key Concept

The geometric method scales much better for larger networks because its complexity depends on the output size (number of EFMs) rather than the combinatorial explosion of input subsets.

### Output

- `notes/demo_efm_methods.png` (default)
- `notes/demo_efm_methods.svg` (with `--svg` flag)

### Run

```bash
python demos/demo_04_efm_methods.py
python demos/demo_04_efm_methods.py --n-max 12  # Adjust max network size
python demos/demo_04_efm_methods.py --svg
```

---

## Visualization Style

All demos use a consistent color palette for publication-quality figures:

| Element | Color | Hex Code |
|---------|-------|----------|
| Thermodynamic space (fill) | Light mint green | `#b9e7b9` or `#e8f5e9` |
| Upper bound | Red/blue | `#c0392b` / `blue` |
| Lower bound | Blue/red | `#2980b9` / `red` |
| Equilibrium line | Green | `#27ae60` |
| Stable steady states | Dark teal | `#2c3e50` |
| Unstable steady states | Gray | `#95a5a6` |
| Scatter points | Dark gray | `#34495e` |

---

## Running All Demos

```bash
cd code/crn

# PNG output (default)
for demo in demos/demo_*.py; do
    echo "Running $demo..."
    python "$demo"
done

# Also generate SVG
for demo in demos/demo_*.py; do
    python "$demo" --svg
done
```

---

## Extending to Custom Networks

```python
import numpy as np
from crn_bounds.api import CRNInput, run_pipeline

# Define your network stoichiometry (species × reactions)
# Example: A ⇌ B, B ⇌ C
Sx = np.array([
    [-1,  0],  # A
    [ 1, -1],  # B
    [ 0,  1],  # C
], dtype=float)

# Standard chemical potentials (μ⁰/RT)
mu0_X = np.zeros(3)

# External driving on first reaction
delta_mu = 2.0
A_Y = np.array([delta_mu, 0.0])

# Compute bounds
result = run_pipeline(
    CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y),
    auto_probes=True
)

# Access results
for i, ab in enumerate(result.affinity_bounds):
    print(f"R{i+1}: A ∈ [{ab.lower:.2f}, {ab.upper:.2f}]")

for i, lb in enumerate(result.reaction_log_bounds):
    print(f"R{i+1}: s·ln(x) ∈ [{lb.lo:.2f}, {lb.hi:.2f}]")
```

---

## Mathematical Background

### Elementary Flux Modes (EFMs)

EFMs are the minimal generators of the steady-state flux cone. For a network with stoichiometry matrix S:
- Steady state: S·J = 0
- Irreversibility: J ≥ 0 (in split representation)
- EFMs are extreme rays of this cone

### Split Representation

Reversible reactions are split into forward/backward pairs:
```
A ⇌ B  →  A → B (forward), B → A (backward)
```

This ensures all fluxes are non-negative, simplifying the polyhedral geometry.

### Affinity Bounds

For EFM e with cycle affinity A_e = e·A_Y:
```
A_ρ^ss ∈ [min(A_e/e_ρ), max(A_e/e_ρ)]
```
where the min/max are over EFMs with e_ρ ≠ 0.

---

## References

1. **Paper**: arXiv:2407.11498 - "Thermodynamic Space of Chemical Reaction Networks"
2. **EFM Algorithms**: See `docs/EFM_ALGORITHMS.md` for detailed algorithm descriptions
3. **pycddlib**: https://github.com/mcmtroffaes/pycddlib
