# Elementary Flux Mode Enumeration: Algorithm Details

This document provides a detailed explanation of the two algorithms implemented in TACOS for enumerating Elementary Flux Modes (EFMs).

## Background

### What are EFMs?

Elementary Flux Modes (EFMs) are minimal sets of reactions that can operate at steady state. Mathematically, an EFM is a vector **e** in the nullspace of the stoichiometric matrix that satisfies:

1. **Steady-state condition**: S<sup>X</sup> **e** = **0**
2. **Minimality**: No proper subset of active reactions can also achieve steady state
3. **Sign consistency**: For irreversible reactions, fluxes must be non-negative

### Split Representation

For networks with reversible reactions, we use a "split" representation where each reversible reaction ρ is replaced by two irreversible reactions (forward and backward):

```
Original: A ⇌ B (reaction ρ)
Split:    A → B (reaction ρ⁺)
          B → A (reaction ρ⁻)
```

The split stoichiometric matrix is:
```
S^{X,split} = [S^X, -S^X] ∈ ℝ^{n_X × 2n_R}
```

The flux cone in split space is:
```
C_split = { w ∈ ℝ^{2n_R}_≥0 | S^{X,split} w = 0 }
```

EFMs correspond to the **extreme rays** of this cone.

---

## Algorithm 1: Combinatorial Method (Circuit Enumeration)

### Concept

The combinatorial method directly searches for **conformal circuits** — minimal subsets of columns in S<sup>X,split</sup> whose nullspace is one-dimensional and spanned by a strictly positive vector.

### Mathematical Foundation

A circuit in linear algebra is a minimal linearly dependent set of vectors. For our flux cone:

- Let r = rank(S<sup>X,split</sup>)
- A circuit has at most r+1 columns (by rank-nullity theorem)
- We search all subsets I of size 2 to r+1

For each subset I, we check:
1. The nullspace of S<sup>X,split</sup>[:,I] is exactly 1-dimensional
2. The null vector has full support on I (no zero entries)
3. All entries can be made non-negative (conformality)

### Algorithm Pseudocode

```python
def enumerate_efms_combinatorial(Sx_split):
    n_X, n_R_split = Sx_split.shape
    rank = matrix_rank(Sx_split)
    efms = []

    # Iterate over subset sizes
    for size in range(2, rank + 2):
        # Iterate over all subsets of columns
        for subset in combinations(range(n_R_split), size):
            sub_matrix = Sx_split[:, subset]

            # Compute SVD to find nullspace
            U, S, Vh = svd(sub_matrix)
            null_dim = count(S < tolerance)

            if null_dim != 1:
                continue  # Not a circuit

            null_vec = Vh[-1, :]  # Last row of Vh

            # Check conformality (all same sign)
            if not (all(null_vec >= 0) or all(null_vec <= 0)):
                continue

            # Check full support
            if count(null_vec != 0) != size:
                continue

            # Embed into full space and add to results
            full_vec = embed(null_vec, subset, n_R_split)
            efms.append(normalize(full_vec))

    return project_to_original_coordinates(efms)
```

### Complexity Analysis

- Number of subsets to check: $\binom{2n_R}{k}$ for each k from 2 to r+1
- For dense matrices, r ≈ n_X, so total subsets ≈ $\sum_{k=2}^{n_X+1} \binom{2n_R}{k}$
- This grows **combinatorially** with network size
- Each subset requires SVD computation: O(n_X · k²)

**Time Complexity**: O(2^{n_R} · n_X · r²) in the worst case

### Advantages
- Conceptually simple
- No external dependencies
- Guaranteed to find all EFMs

### Disadvantages
- Exponential scaling with network size
- Impractical for networks with >10 reactions

---

## Algorithm 2: Geometric Method (Polyhedral Extreme Ray Enumeration)

### Concept

The geometric method treats the flux cone as a **polyhedral cone** and uses algorithms from computational geometry to enumerate its extreme rays.

### Mathematical Foundation

The flux cone C_split can be represented in two equivalent forms:

**H-representation** (halfspace intersection):
```
C_split = { w | S^{X,split} w = 0, w ≥ 0 }
```

**V-representation** (vertex/ray enumeration):
```
C_split = cone(e_1, e_2, ..., e_m)
```
where e_i are the extreme rays (EFMs).

The geometric method converts from H-representation to V-representation using the **Double Description Method** (Motzkin et al., 1953).

### Double Description Method

The algorithm maintains a list of generators (rays) and iteratively adds constraints:

1. Start with the cone defined by w ≥ 0 (generators are coordinate axes)
2. For each equality constraint (row of S<sup>X,split</sup>):
   - Classify existing rays as:
     - **Zero**: satisfies the constraint
     - **Positive**: violates in positive direction
     - **Negative**: violates in negative direction
   - Keep all zero rays
   - Generate new rays from positive-negative pairs
   - Discard pure positive and pure negative rays

### Algorithm Pseudocode

```python
def enumerate_efms_geometric(Sx_split):
    n_X, n_R_split = Sx_split.shape

    # Build H-representation matrix for cddlib
    # Format: [b | A] where Ax ≤ b (or Ax = b for equalities)
    H = build_h_matrix(Sx_split)

    # Mark equality constraints (steady-state)
    equalities = range(n_X)

    # Convert to cdd matrix
    mat = cdd.matrix_from_array(H, rep_type=INEQUALITY)
    mat.lin_set = frozenset(equalities)

    # Compute V-representation (extreme rays)
    poly = cdd.polyhedron_from_matrix(mat)
    generators = cdd.copy_generators(poly)

    # Extract rays (not vertices)
    efms_split = []
    for row in generators:
        if row[0] == 0:  # Ray indicator
            ray = row[1:]
            if norm(ray) > tolerance:
                efms_split.append(ray)

    return project_to_original_coordinates(efms_split)
```

### Implementation with pycddlib

We use `pycddlib`, a Python wrapper for the `cddlib` library which implements the Double Description Method with exact rational arithmetic.

```python
import cdd

def enumerate_extreme_rays(Sx_split):
    n_X, n_R_split = Sx_split.shape

    # Build constraint matrix:
    # [0 | Sx_split]  for equalities (Sx_split @ w = 0)
    # [0 | I]         for non-negativity (w >= 0)

    n_eq = n_X
    n_ineq = n_R_split

    H = np.zeros((n_eq + n_ineq, 1 + n_R_split))
    H[:n_eq, 1:] = Sx_split        # Equalities
    H[n_eq:, 1:] = np.eye(n_R_split)  # w >= 0

    mat = cdd.matrix_from_array(H, rep_type=cdd.RepType.INEQUALITY)
    mat.lin_set = frozenset(range(n_eq))

    poly = cdd.polyhedron_from_matrix(mat)
    gens = cdd.copy_generators(poly)

    rays = [np.array(row[1:]) for row in gens.array if row[0] == 0]
    return rays
```

### Complexity Analysis

The Double Description Method is **output-sensitive**:

- Let m = number of extreme rays (EFMs)
- Let n = number of constraints (n_X + 2n_R)
- Time complexity: O(n · m² · d) where d is dimension

**Key insight**: The runtime scales with the *number of EFMs found*, not with the total number of possible subsets.

For networks where the number of EFMs grows moderately, this is dramatically faster than the combinatorial method.

### Advantages
- Output-sensitive complexity
- Handles larger networks (up to ~20-30 reactions)
- Exact arithmetic available (avoids numerical issues)

### Disadvantages
- Requires external library (cddlib)
- Still exponential in worst case (EFM count can be exponential)

---

## Comparison: Benchmarks on Modular Self-Assembly Network

We benchmark both methods on a parameterized self-assembly network where:
- Target structure size n determines network complexity
- Number of reactions = 2n
- Number of EFMs grows exponentially with n

### Network Structure

```
Reactions:
  R0: F + A → B₁ + W      (activation)
  R1: B₁ + B₁ → B₂        (dimerization)
  R2: B₁ + B₂ → B₃        (assembly)
  ...
  Rₙ₋₁: B₁ + Bₙ₋₁ → Bₙ   (final assembly)
  Rₙ: B₁ → A              (degradation)
  Rₙ₊₁: B₂ → 2A           (degradation)
  ...
  R₂ₙ₋₁: Bₙ → nA          (degradation)
```

### Benchmark Results

| n | Reactions | EFMs | Combinatorial (s) | Geometric (s) |
|---|-----------|------|-------------------|---------------|
| 3 | 6 | 3 | 0.046 | 0.008 |
| 4 | 8 | 8 | 0.243 | 0.002 |
| 5 | 10 | 18 | 1.659 | 0.003 |
| 6 | 12 | 36 | 15.67 | 0.012 |
| 7 | 14 | 66 | - | 0.036 |
| 8 | 16 | 113 | - | 0.104 |
| 9 | 18 | 183 | - | 0.270 |
| 10 | 20 | 283 | - | 0.637 |
| 12 | 24 | 606 | - | 2.929 |

(-) indicates timeout (>30 seconds)

### Key Observations

1. **Exponential growth of EFMs**: The number of EFMs grows approximately as O(1.5^n)

2. **Combinatorial explosion**: The combinatorial method becomes impractical beyond n=6

3. **Geometric efficiency**: The geometric method remains usable up to n=15+

4. **Scaling behavior**:
   - Combinatorial: ~O(2^n) time
   - Geometric: ~O(m²) where m = number of EFMs

---

## When to Use Which Method

| Scenario | Recommended Method |
|----------|-------------------|
| Small networks (≤6 reactions) | Either (combinatorial simpler to understand) |
| Medium networks (7-20 reactions) | Geometric |
| Large networks (>20 reactions) | Geometric + specialized solvers |
| Teaching/debugging | Combinatorial |
| Production use | Geometric |

---

## References

1. Schuster, S., & Hilgetag, C. (1994). On elementary flux modes in biochemical reaction systems at steady state. *Journal of Biological Systems*, 2(02), 165-182.

2. Fukuda, K., & Prodon, A. (1996). Double description method revisited. *Combinatorics and Computer Science*, 91-111.

3. Terzer, M., & Stelling, J. (2008). Large-scale computation of elementary flux modes with bit pattern trees. *Bioinformatics*, 24(19), 2229-2235.

4. Müller, S., & Bockmayr, A. (2016). Elementary vectors and autocatalytic sets for resource allocation in next-generation models of cellular growth. *bioRxiv*, 078998.
