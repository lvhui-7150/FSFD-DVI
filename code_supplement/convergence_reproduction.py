"""Numba-accelerated reproduction of the FSFD-DVI convergence study.

The solver implements the scalar crisp-singleton model in the manuscript
(Eq. for the numerical experiments) with the same explicit L1-Euler-Maruyama
scheme and the same projected-gradient inner solve as
``revised_paper_figures.py``.  The delay is commensurate for every grid in the
study (tau = 0.3 and h = T/N with T = 0.75), so the delayed state is read
directly from the history array.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


ROOT = Path(__file__).resolve().parents[1]


@njit
def _solve_crisp_control(y: float, y_tau: float, k_prev: float,
                         gamma: float) -> float:
    """Projected-gradient solve of the crisp VI on K=[-1.2,1.2]."""
    k = k_prev
    rho = 0.05
    for _ in range(40):
        grad = k + 0.5 * y - gamma * y_tau
        k_new = max(-1.2, min(1.2, k - rho * grad))
        if abs(k_new - k) < 1e-6:
            break
        k = k_new
    return k


@njit
def _simulate_crisp(p: float, T: float, tau: float, gamma: float,
                    sigma: float, z: np.ndarray) -> np.ndarray:
    """Simulate one path with commensurate delay and standard-normal increments."""
    N = z.shape[0]
    h = T / N
    l = int(round(tau / h))
    cp = math.exp(math.lgamma(2.0 - p))

    b = np.empty(N + 2)
    for j in range(N + 2):
        b[j] = (j + 1) ** (1.0 - p) - j ** (1.0 - p)

    start = l + 1
    y_all = np.empty(start + N + 1)
    k_all = np.zeros(start + N + 1)
    for i in range(start + 1):
        y_all[i] = 1.0

    c_det = cp * h ** p
    c_noise = cp * h ** (p - 0.5)

    for m in range(N):
        n = start + m
        y = y_all[n]
        y_tau = y_all[n - l]
        k = _solve_crisp_control(y, y_tau, k_all[n - 1], gamma)
        k_all[n] = k

        memory = 0.0
        for j in range(1, m + 1):
            memory += b[j] * (y_all[n + 1 - j] - y_all[n - j])

        drift = -1.8 * y + 0.3 * y_tau + k
        y_all[n + 1] = (y - memory + c_det * drift
                        + c_noise * sigma * y * z[m])

    return y_all[start:start + N + 1]


def run_convergence(runs: int = 200, seed: int = 999,
                    T: float = 0.75, p: float = 0.75,
                    gamma: float = 0.5, tau: float = 0.3,
                    sigma: float = 0.1, N_ref: int = 8000,
                    ns: tuple[int, ...] = (20, 40, 80, 160, 320)):
    """Run the pathwise-coupled fine/coarse convergence study."""
    rng = np.random.default_rng(seed)
    errors = {n: np.empty(runs) for n in ns}

    for r in range(runs):
        z_ref = rng.standard_normal(N_ref)
        y_ref = _simulate_crisp(p, T, tau, gamma, sigma, z_ref)
        for n in ns:
            q = N_ref // n
            z_coarse = z_ref.reshape(n, q).sum(axis=1) / math.sqrt(q)
            y_n = _simulate_crisp(p, T, tau, gamma, sigma, z_coarse)
            errors[n][r] = math.sqrt(np.mean((y_n - y_ref[::q]) ** 2))

    hs = np.asarray([T / n for n in ns])
    means = np.asarray([float(np.mean(errors[n])) for n in ns])
    stds = np.asarray([float(np.std(errors[n], ddof=1)) for n in ns])
    ses = stds / math.sqrt(runs)
    log_h = np.log(hs)
    log_e = np.log(means)
    fit = np.polyfit(log_h, log_e, 1)
    slope = float(fit[0])

    resid = log_e - np.polyval(fit, log_h)
    dof = len(ns) - 2
    se_slope = float(np.sqrt(np.sum(resid ** 2) / dof
                             / np.sum((log_h - np.mean(log_h)) ** 2)))
    lo = slope - 1.96 * se_slope
    hi = slope + 1.96 * se_slope

    return {
        "runs": runs,
        "seed": seed,
        "N_ref": N_ref,
        "T": T,
        "p": p,
        "gamma": gamma,
        "tau": tau,
        "sigma": sigma,
        "ns": list(ns),
        "hs": hs.tolist(),
        "rmse": means.tolist(),
        "rmse_se": ses.tolist(),
        "slope": slope,
        "slope_se": se_slope,
        "slope_ci": [lo, hi],
        "errors": {str(n): errors[n].tolist() for n in ns},
    }


def render_figure(result: dict, output: Path) -> None:
    ns = result["ns"]
    hs = np.asarray(result["hs"])
    means = np.asarray(result["rmse"])
    slope = result["slope"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(ns, means, "D-", label="Measured RMSE", markersize=6)
    fit_line = means[0] * (ns[0] / np.asarray(ns)) ** slope
    ax.loglog(ns, fit_line, "k--", label=rf"Fit slope $={slope:.3f}$")
    ax.set_xlabel("Steps (N)")
    ax.set_ylabel("RMSE")
    ax.set_title("Convergence Rate Analysis")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    result = run_convergence(runs=args.runs, seed=args.seed)
    out = ROOT / "code_supplement" / "convergence_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    text = ROOT / "code_supplement" / "convergence_slope.txt"
    text.write_text(
        f"slope={result['slope']:.4f}\n"
        f"runs={result['runs']}\n"
        f"seed={result['seed']}\n"
        f"N_ref={result['N_ref']}\n",
        encoding="utf-8",
    )

    fig = ROOT / "fig" / "Fig6_Convergence_Rate.png"
    render_figure(result, fig)

    print(json.dumps({k: result[k] for k in result if k != "errors"}, indent=2))


if __name__ == "__main__":
    main()
