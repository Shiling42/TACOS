#!/usr/bin/env python3
"""
TACOS Demo 3: Chiral Symmetry Breaking (Frank Model)
=====================================================

Network:
  R1: R + A ⇌ 2R   (R autocatalysis)
  R2: S + A ⇌ 2S   (S autocatalysis)
  R3: R + S ⇌ C    (mutual inhibition)

Parameters from paper: k0+ = 1, k0- = 0.7, k1+ = 16, k1- = 2, c = 0.5

Thermodynamic bounds (Eq. 27):
  -Δμ/RT ≤ ln(r^ss/s^ss) ≤ +Δμ/RT

IMPORTANT: Δμ includes standard potential contribution from rate ratios:
  Δμ/RT = (2μ°_A - μ°_C)/RT + ln(a²/c)
        = -2 ln(k0+/k0-) - ln(k1+/k1-) + ln(a²/c)

Reference: arXiv:2407.11498, Fig. chiral_breaking
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

COLORS = {
    'bound_fill': '#b9e7b9',
    'upper_bound': 'blue',
    'lower_bound': 'red',
    'stable_R': '#e74c3c',
    'stable_S': '#3498db',
    'racemic': '#2c3e50',
    'unstable': '#95a5a6',
}

# Parameters from paper
K0P, K0M = 1.0, 0.7
K1P, K1M = 16.0, 2.0
C = 0.5

# Standard potential contribution: (2μ°_A - μ°_C)/RT
# From LDB: μ°_R - μ°_A = -RT ln(k0+/k0-), μ°_C - 2μ°_R = -RT ln(k1+/k1-)
# => (2μ°_A - μ°_C)/RT = 2 ln(k0+/k0-) + ln(k1+/k1-)
DELTA_MU_STD = 2 * np.log(K0P / K0M) + np.log(K1P / K1M)


def compute_delta_mu(a):
    """Compute the CORRECT Δμ/RT including standard potentials."""
    return DELTA_MU_STD + np.log(a**2 / C)


def chiral_ode(t, y, a):
    """Frank model ODEs."""
    r, s = y
    dr = K0P * a * r - K0M * r**2 - K1P * r * s + K1M * C
    ds = K0P * a * s - K0M * s**2 - K1P * r * s + K1M * C
    return [dr, ds]


def find_steady_states(a, n_init=80):
    """Find steady states by integration from random ICs."""
    states = []
    for _ in range(n_init):
        r0 = np.exp(np.random.uniform(-3, 3))
        s0 = np.exp(np.random.uniform(-3, 3))
        try:
            sol = solve_ivp(lambda t, y: chiral_ode(t, y, a), [0, 2000],
                            [r0, s0], method='LSODA', rtol=1e-10, atol=1e-12)
            if sol.success:
                r_ss, s_ss = sol.y[:, -1]
                if r_ss > 1e-10 and s_ss > 1e-10:
                    rhs = chiral_ode(0, [r_ss, s_ss], a)
                    if np.linalg.norm(rhs) < 1e-6:
                        ln_ratio = np.log(r_ss / s_ss)
                        is_new = all(abs(s['ln_ratio'] - ln_ratio) > 0.1 for s in states)
                        if is_new:
                            states.append({'r': r_ss, 's': s_ss, 'ln_ratio': ln_ratio})
        except:
            pass
    return states


def check_stability(r_ss, s_ss, a):
    """Check stability via Jacobian eigenvalues."""
    eps = 1e-8
    y0 = [r_ss, s_ss]
    f0 = chiral_ode(0, y0, a)
    J = np.zeros((2, 2))
    for j in range(2):
        y_pert = y0.copy()
        y_pert[j] += eps
        f_pert = chiral_ode(0, y_pert, a)
        J[:, j] = (np.array(f_pert) - np.array(f0)) / eps
    eigvals = np.linalg.eigvals(J)
    return np.all(np.real(eigvals) < 0)


def compute_bifurcation(a_values):
    """Compute pitchfork bifurcation diagram."""
    np.random.seed(42)
    results = []
    for a in a_values:
        delta_mu = compute_delta_mu(a)
        states = find_steady_states(a)
        for s in states:
            stable = check_stability(s['r'], s['s'], a)
            results.append({
                'a': a,
                'delta_mu': delta_mu,
                'ln_ratio': s['ln_ratio'],
                'stable': stable,
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='notes/demo_chiral.png')
    parser.add_argument('--svg', action='store_true', help='Also save SVG')
    args = parser.parse_args()

    outdir = Path(args.out).parent
    outdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("TACOS Demo 3: Chiral Symmetry Breaking")
    print("=" * 60)
    print("\nFrank model: R+A ⇌ 2R, S+A ⇌ 2S, R+S ⇌ C")
    print(f"Parameters: k0+={K0P}, k0-={K0M}, k1+={K1P}, k1-={K1M}, c={C}")
    print(f"\nStandard potential contribution: {DELTA_MU_STD:.3f} RT")
    print("Full driving: Δμ/RT = {:.3f} + ln(a²/c)".format(DELTA_MU_STD))

    # Compute bifurcation
    a_values = np.linspace(0.2, 8.0, 80)
    bif_data = compute_bifurcation(a_values)

    stable = [d for d in bif_data if d['stable']]
    unstable = [d for d in bif_data if not d['stable']]
    print(f"\nFound {len(stable)} stable, {len(unstable)} unstable points")

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Thermodynamic bounds: ln(r/s) ∈ [-Δμ/RT, +Δμ/RT]
    dm_range = np.linspace(0, 8, 200)
    ax.fill_between(dm_range, -dm_range, dm_range, color=COLORS['bound_fill'],
                    alpha=0.4, label='Thermodynamic space')
    ax.plot(dm_range, dm_range, color=COLORS['upper_bound'], lw=2.3)
    ax.plot(dm_range, -dm_range, color=COLORS['lower_bound'], lw=2.3)

    # Bifurcation data
    if stable:
        dms = [d['delta_mu'] for d in stable]
        ratios = [d['ln_ratio'] for d in stable]
        colors = [COLORS['stable_R'] if r > 0.1 else
                  COLORS['stable_S'] if r < -0.1 else
                  COLORS['racemic'] for r in ratios]
        ax.scatter(dms, ratios, c=colors, s=12, alpha=0.8, zorder=5)

        # Legend entries
        ax.scatter([], [], c=COLORS['stable_R'], s=30, label='R-dominant')
        ax.scatter([], [], c=COLORS['stable_S'], s=30, label='S-dominant')
        ax.scatter([], [], c=COLORS['racemic'], s=30, label='Racemic')

    if unstable:
        ax.scatter([d['delta_mu'] for d in unstable],
                   [d['ln_ratio'] for d in unstable],
                   c=COLORS['unstable'], s=8, alpha=0.5, marker='x',
                   label='Unstable', zorder=4)

    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)

    ax.set_xlabel(r'Driving $\Delta\mu/RT$', fontsize=11)
    ax.set_ylabel(r'$\ln(r^{\rm ss}/s^{\rm ss})$', fontsize=11)
    ax.set_title('Chiral Symmetry Breaking: Pitchfork Bifurcation', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 8)
    ax.set_ylim(-8, 8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.out}")

    if args.svg:
        svg_out = args.out.replace('.png', '.svg')
        plt.savefig(svg_out, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_out}")

    plt.close()

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
• Pitchfork bifurcation: racemic → chiral states
• Chiral branches stay WITHIN thermodynamic bounds ±Δμ/RT
• Perfect homochirality requires infinite Δμ
• Key: Δμ includes standard potential from rate ratios!
  Δμ/RT = {:.2f} + ln(a²/c), NOT just ln(a²/c)
""".format(DELTA_MU_STD))


if __name__ == '__main__':
    main()
