from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Any, Tuple

try:  # numpy is optional to allow importing bs_call_price without it
    import numpy as np  # type: ignore
    from numpy.typing import NDArray  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    np = None  # type: ignore
    NDArray = Any  # type: ignore


def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black–Scholes European call; handles sigma=0 or T=0 edge by discounted intrinsic."""
    if T <= 0.0 or sigma <= 0.0:
        return exp(-r * max(T, 0.0)) * max(S0 - K, 0.0)
    vsqrt = sigma * sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    from math import erf

    def phi(x: float) -> float:
        # standard normal CDF via erf
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    return S0 * phi(d1) - K * exp(-r * T) * phi(d2)


@dataclass(frozen=True)
class MCResult:
    price: float
    se: float
    ci95: Tuple[float, float]
    n_effective: int


def call_mc_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    nsim: int = 200_000,
    seed: int | None = 123,
    antithetic: bool = True,
) -> MCResult:
    """Risk-neutral MC for a European call with optional antithetic variates."""
    if np is None:  # pragma: no cover - depends on optional numpy
        raise ImportError("NumPy is required for call_mc_price")
    assert nsim > 0
    rng = np.random.default_rng(seed)
    if antithetic:
        m = (nsim + 1) // 2  # half, round up
        Z = rng.standard_normal(m)
        drift = (r - 0.5 * sigma * sigma) * T
        vol = sigma * sqrt(T)
        s_plus: NDArray[np.float64] = S0 * np.exp(drift + vol * Z)
        s_minus: NDArray[np.float64] = S0 * np.exp(drift - vol * Z)
        payoff = 0.5 * (np.maximum(s_plus - K, 0.0) + np.maximum(s_minus - K, 0.0))
        # If nsim is odd, we used m = ceil(n/2); effective N is m, but variance reflects both halves
        n_eff = m
    else:
        Z = rng.standard_normal(nsim)
        drift = (r - 0.5 * sigma * sigma) * T
        vol = sigma * sqrt(T)
        ST = S0 * np.exp(drift + vol * Z)
        payoff = np.maximum(ST - K, 0.0)
        n_eff = nsim

    disc = exp(-r * T)
    disc_payoff = disc * payoff
    price = float(np.mean(disc_payoff))
    std = float(np.std(disc_payoff, ddof=1))
    se = std / sqrt(n_eff)
    ci = (price - 1.96 * se, price + 1.96 * se)
    return MCResult(price=price, se=se, ci95=ci, n_effective=n_eff)


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    res = call_mc_price(S0, K, r, sigma, T, nsim=1_000_000, seed=42, antithetic=True)
    bs = bs_call_price(S0, K, r, sigma, T)
    print(
        f"MC={res.price:.5f}  SE={res.se:.5f}  CI95=[{res.ci95[0]:.5f}, {res.ci95[1]:.5f}]  "
        f"N_eff={res.n_effective}  BS={bs:.5f}"
    )
