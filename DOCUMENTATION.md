# CRN Bounds — Documentation / Development Record

This document summarizes what was implemented, how the pieces fit together, and how to reproduce the paper examples.

Repository workspace root (OpenClaw runtime):
- `/root/.openclaw/workspace/`

Project root:
- `/root/.openclaw/workspace/code/crn/`

Paper materials used:
- `/root/.openclaw/workspace/papers/arxiv-2407.11498/`

> All thermodynamic quantities are treated **dimensionless** (divided by RT).

---

## 1. Goal

Implement a small, test-driven toolkit that can reproduce the key *thermodynamic bounds* of arXiv:2407.11498:

- Reaction affinity bounds at steady state from EFMs
- Chemical-probe bounds (affinity + concentration/log-ratio)
- Reproduction of paper examples:
  - Dissipative self-assembly (including probe)
  - Schlögl model (affinity + concentration band shape)
  - Chiral symmetry breaking (Frank model + probe)

Additionally, implement a *validation simulation* workflow:

- sample rates consistent with local detailed balance (LDB)
- integrate mass-action ODEs to steady state
- plot steady-state points as scatter inside the thermodynamic bands

---

## 2. Main API

### 2.1 CRNInput

`crn_bounds/api.py` defines:

```python
@dataclass(frozen=True)
class CRNInput:
    Sx: np.ndarray  # internal stoichiometry (nX x nR)
    mu0_X: np.ndarray  # standard chemical potentials for internal species (nX,)
    A_Y: np.ndarray  # external affinity contributions per reaction (nR,)
```

Conventions:
- `Sx[:,rho]` is the internal stoichiometry vector of reaction `rho`.
- `mu0_X` is in units of RT.
- `A_Y[rho]` is the external/chemostat contribution to affinity of reaction `rho` (units RT).

### 2.2 run_pipeline()

```python
res = run_pipeline(
    CRNInput(Sx=Sx, mu0_X=mu0_X, A_Y=A_Y),
    auto_probes=True,
    max_probes=10,
)
```

`PipelineResult` includes:

- `efms_split`: extreme rays in split-forward/backward variable space
- `efms_orig`: EFMs mapped back into original reaction coordinate system
- `affinity_bounds`: list of `[lower, upper]` for each reaction’s affinity at steady state
- `reaction_log_bounds`: list of `LogRatioBound` objects (one per reaction)
- `probes`: list of `ProbeResult` (auto-generated probes), each with:
  - `probe_sx`
  - `affinity_bound`
  - `log_ratio_bound`

---

## 3. Core algorithms

### 3.1 Split reversible reactions

File: `crn_bounds/split.py`

To enumerate EFMs with `cddlib`, we use a standard trick:
- represent each reversible reaction `v_rho` as the difference of nonnegative variables `(v_rho^+ - v_rho^-)`.
- build an augmented stoichiometry matrix for the split variables.

This produces a nonnegative cone suitable for extreme-ray enumeration.

### 3.2 Extreme ray / EFM enumeration

File: `crn_bounds/efm.py`

Uses `pycddlib` (cddlib) to enumerate extreme rays of the cone defined by `Sx_split v >= 0` constraints.

We then map rays back to original coordinates using the split mapping (`split_to_orig`, `split_sign`).

### 3.3 Affinity bounds

File: `crn_bounds/affinity.py`

Given:
- EFMs `E` in original coordinates
- external driving vector `A_Y`

we compute steady-state bounds on reaction affinities by normalizing EFMs and taking extremal values.

This reproduces the paper’s EFM-based affinity bounds (Eq.20/Eq.24 structure in the paper).

### 3.4 Concentration (log-ratio) bounds

File: `crn_bounds/concentration_bounds.py`

We represent concentration constraints as:

```text
lo <= s^T ln x <= hi
```

in a `LogRatioBound(s, lo, hi)` object.

Implemented:

- `probe_log_ratio_bound(...)`:
  - for an *added* internal probe (external affinity 0)
  - uses extended-network EFMs to bound probe affinity
  - then translates into a bound on the probe’s concentration log-ratio

- `reaction_log_ratio_bound(...)`:
  - currently targeted at reproducing the paper’s showcased self-assembly ratios
  - returns `LogRatioBound` for each original reaction `rho`

> Note: A fully general Π-pathway construction for concentration bounds across arbitrary CRNs is a next-step extension.

### 3.5 Auto-probes from conservation laws

Files:
- `crn_bounds/conservation.py`
- `crn_bounds/probes.py`

We compute integer conservation laws and generate a set of small-integer candidate probe stoichiometries.

Each probe is appended as an additional reaction and EFMs are re-enumerated to produce:
- probe affinity bounds
- probe concentration log-ratio bounds

---

## 4. Simulation workflow (scatter validation)

### 4.1 LDB-consistent rate sampling

File: `crn_bounds/ldb.py`

We sample mass-action forward/backward rates consistent with local detailed balance:
- choose random prefactors
- enforce `k^+/k^-` ratios consistent with `ΔG° + μ_Y` and stoichiometry

### 4.2 ODE relaxation

File: `crn_bounds/model.py`

We integrate ODEs under mass-action kinetics to a steady state using a simple relaxation integrator.

### 4.3 Self-assembly multi-panel scatter plots

File: `code/crn/scripts/sample_self_assembly_relaxation_panels.py`

Runs *one batch* of simulations and reuses the same steady-state points across multiple thermodynamic panels:

- R1 ratio: `ln(x2)` vs `ln(x1^2)`
- R2 ratio: `ln(x3)` vs `ln(x1 x2)`
- R3 ratio: `ln(x1^3)` vs `ln(x3)`
- Probe ratio: `ln(x3^2)` vs `ln(x2^3)`

Important fix:
- probe band uses `ln(x3^2/x2^3) in [-3Δμ, 0]` (not `[-Δμ, 0]`).

The points are also saved to:
- `papers/arxiv-2407.11498/notes/self_assembly_relax_points.npz`

---

## 5. Reproducing paper examples

### 5.1 Self-assembly

Inputs (internal stoichiometry block):

```
Sx = [[-2, -1,  3],
      [ 1, -1,  0],
      [ 0,  1, -1]]
```

Driving:
- set `A_Y=[Δμ,0,0]` (dimensionless)

Probe:
- auto-probe finds `s=[0,-3,2]` among candidates
- probe ratio bound matches paper: `ln(x3^2/x2^3) in [-3Δμ, 0]`

Tests:
- `tests/test_api_self_assembly_probes.py`
- `tests/test_api_self_assembly_conc.py`

Key generated figures (notes folder):
- `fig_self_assembly_from_efm.png`
- `fig_self_assembly_relax_panel_R1.png`
- `fig_self_assembly_relax_panel_R2.png`
- `fig_self_assembly_relax_panel_R3.png`
- `fig_self_assembly_relax_panel_probe.png`

### 5.2 Schlögl model

We reproduce the bound band implied by paper Eq. con_bound_Schlogl.

- plotting script: `scripts/plot_schlogl_bounds.py`
- test for affinity structure: `tests/test_api_schlogl.py`

Figure:
- `notes/fig_schlogl_thermo_space.png`

### 5.3 Chiral symmetry breaking (Frank model)

We reproduce the theoretical probe bound:

- `ln(r/s) in [-Δμ, +Δμ]`

- test: `tests/test_api_frank_probe.py`
- plotting script: `scripts/plot_chiral_bounds.py`

Figure:
- `notes/fig_chiral_bound.png`

---

## 6. How to run everything

From workspace root:

```bash
cd /root/.openclaw/workspace

# run tests
PYTHONPATH=code/crn pytest -q

# generate self-assembly scatter panels (600 points)
PYTHONPATH=code/crn python3 code/crn/scripts/sample_self_assembly_relaxation_panels.py \
  --outdir papers/arxiv-2407.11498/notes --DeltaMu_over_RT 2.0 --n 600 --seed 0

# generate Schlögl and chiral bound plots
PYTHONPATH=code/crn python3 code/crn/scripts/plot_schlogl_bounds.py \
  --out papers/arxiv-2407.11498/notes/fig_schlogl_thermo_space.png

PYTHONPATH=code/crn python3 code/crn/scripts/plot_chiral_bounds.py \
  --out papers/arxiv-2407.11498/notes/fig_chiral_bound.png
```

---

## 7. Delivered artifacts (paths)

- Code:
  - `/root/.openclaw/workspace/code/crn/`

- Notes + generated figures:
  - `/root/.openclaw/workspace/papers/arxiv-2407.11498/notes/`

---

## 8. Changelog highlights (high level)

- Implemented EFM enumeration via cddlib
- Implemented affinity bounds + probe bounds
- Added concentration log-ratio bounds for probes and self-assembly reactions
- Added LDB-consistent sampling + relaxation simulations and scatter plots
- Added tests for self-assembly, Schlögl, and chiral symmetry breaking examples

