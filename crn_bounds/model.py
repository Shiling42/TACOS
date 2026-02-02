"""General CRN model utilities: mass-action kinetics, chemostats, steady states.

This module is for validating the paper's bounds by sampling random kinetics
and checking that steady states fall inside predicted thermodynamic space.

We provide:
- MassActionCRN container (stoichiometry split into reactants/products)
- flux evaluation (net + forward/backward)
- steady state (root) helper
- relaxation (ODE integration) helper (preferred for validation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class MassActionCRN:
    # internal species
    nuX_plus: NDArray[np.float64]   # (nX, nR)
    nuX_minus: NDArray[np.float64]  # (nX, nR)
    # external/chemostatted species (optional)
    nuY_plus: NDArray[np.float64] | None = None   # (nY, nR)
    nuY_minus: NDArray[np.float64] | None = None  # (nY, nR)

    def Sx(self) -> NDArray[np.float64]:
        return self.nuX_minus - self.nuX_plus


def mass_action_flux_split(
    x: NDArray[np.float64],
    crn: MassActionCRN,
    k_plus: NDArray[np.float64],
    k_minus: NDArray[np.float64],
    y: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (net, forward, backward) flux vectors."""
    x = np.asarray(x, dtype=float)

    log_x = np.log(np.clip(x, 1e-300, None))
    log_fwd = np.log(np.clip(k_plus, 1e-300, None)).copy()
    log_bwd = np.log(np.clip(k_minus, 1e-300, None)).copy()

    log_fwd += (crn.nuX_plus.T @ log_x)
    log_bwd += (crn.nuX_minus.T @ log_x)

    if crn.nuY_plus is not None:
        if y is None:
            raise ValueError("y must be provided when nuY_plus is not None")
        log_y = np.log(np.clip(y, 1e-300, None))
        log_fwd += (crn.nuY_plus.T @ log_y)
        log_bwd += (crn.nuY_minus.T @ log_y)

    fwd = np.exp(log_fwd)
    bwd = np.exp(log_bwd)
    net = fwd - bwd
    return net, fwd, bwd


def steady_state(
    crn: MassActionCRN,
    k_plus: NDArray[np.float64],
    k_minus: NDArray[np.float64],
    *,
    y: NDArray[np.float64] | None = None,
    x0: NDArray[np.float64] | None = None,
    method: str = "hybr",
    conservation_r: NDArray[np.float64] | None = None,
    conservation_total: float | None = None,
) -> NDArray[np.float64]:
    """Solve Sx * J(x)=0 for x>0 using root finding in log-space.

    NOTE: For validation we prefer relaxation() below.
    """
    Sx = crn.Sx()
    nX = Sx.shape[0]
    if x0 is None:
        x0 = np.ones(nX)

    if (conservation_r is None) ^ (conservation_total is None):
        raise ValueError("Provide both conservation_r and conservation_total, or neither")

    def f(z: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.exp(np.clip(z, -50, 50))
        J, _, _ = mass_action_flux_split(x, crn, k_plus, k_minus, y=y)
        eq = Sx @ J
        if conservation_r is None:
            return eq
        g = np.empty_like(eq)
        g[:-1] = eq[:-1]
        g[-1] = float(conservation_r @ x) - float(conservation_total)
        return g

    z0 = np.log(np.clip(x0, 1e-12, None))
    res = root(f, z0, method=method)
    if not res.success:
        raise RuntimeError(f"steady_state root fail: {res.message}")

    return np.exp(np.clip(res.x, -50, 50))


def relaxation(
    crn: MassActionCRN,
    k_plus: NDArray[np.float64],
    k_minus: NDArray[np.float64],
    *,
    y: NDArray[np.float64] | None = None,
    x0: NDArray[np.float64],
    t_final: float = 200.0,
    n_eval: int = 2000,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> NDArray[np.float64]:
    """Relax ODE dx/dt = Sx * J(x) from x0 to a late-time state.

    This respects conservation laws automatically (numerically).
    """
    Sx = crn.Sx()

    def rhs(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.clip(x, 1e-300, None)
        J, _, _ = mass_action_flux_split(x, crn, k_plus, k_minus, y=y)
        return Sx @ J

    t_eval = np.linspace(0.0, t_final, n_eval)
    sol = solve_ivp(rhs, (0.0, t_final), x0, t_eval=t_eval, method="LSODA", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"relaxation failed: {sol.message}")

    xT = sol.y[:, -1]
    return np.clip(xT, 1e-300, None)
