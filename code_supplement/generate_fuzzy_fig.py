"""Generate Fig10: effect of the fuzzy alpha-level on system dynamics.

The fuzzy VI is reduced exactly as in the manuscript: at each node we form
the alpha-level interval

    [H_{(y,z,k)}]_alpha = [r - delta(2-alpha), r + delta(2-alpha)],

take the unique minimal-norm element of that interval, and solve the induced
classical variational inequality by projected gradient (Algorithm 1).  All
alpha values use the same 50 Brownian paths.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def simulate(p, tau, gamma_val, sigma_val, T, N, alpha_level, delta, z):
    """Simulate one path for the minimal-norm reduced fuzzy VI."""
    h = T / N
    l = int(round(tau / h))
    cp = math.gamma(2.0 - p)

    b = np.array([(j + 1) ** (1.0 - p) - j ** (1.0 - p)
                  for j in range(1, N + 1)])
    start = l + 1
    y_all = np.ones(start + N + 1)
    k_all = np.zeros(start + N + 1)
    spread = delta * (2.0 - alpha_level)

    for m in range(N):
        n = start + m
        y = y_all[n]
        y_tau = y_all[n - l]

        k = k_all[n - 1]
        a = 0.5 * y - gamma_val * y_tau
        for _ in range(40):
            lo = k + a - spread
            hi = k + a + spread
            if lo > 0.0:
                grad = lo
            elif hi < 0.0:
                grad = hi
            else:
                grad = 0.0
            k_new = max(-1.2, min(1.2, k - 0.05 * grad))
            if abs(k_new - k) < 1e-6:
                break
            k = k_new
        k_all[n] = k

        memory = 0.0
        for j in range(1, m + 1):
            memory += b[j - 1] * (y_all[n + 1 - j] - y_all[n - j])

        drift = -1.8 * y + 0.3 * y_tau + k
        y_all[n + 1] = (
            y
            - memory
            + cp * h ** p * drift
            + cp * h ** (p - 0.5) * sigma_val * y * z[m]
        )

    return y_all[start:start + N + 1]


def main():
    p = 0.75
    tau = 0.3
    gamma_val = 0.5
    sigma_val = 0.1
    T = 3.0
    N = 500
    delta = 0.3
    n_paths = 50
    seed = 42

    rng = np.random.default_rng(seed)
    z_paths = rng.standard_normal((n_paths, N))

    alpha_values = [0.3, 0.6, 0.9, 1.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = [r"$\alpha=0.3$", r"$\alpha=0.6$",
              r"$\alpha=0.9$", r"$\alpha=1.0$"]
    t = np.linspace(0, T, N + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha_val, color, label in zip(alpha_values, colors, labels):
        ys = [
            simulate(p, tau, gamma_val, sigma_val, T, N, alpha_val, delta, z)
            for z in z_paths
        ]
        ax.plot(t, np.mean(np.asarray(ys), axis=0),
                color=color, linewidth=1.8, label=label)

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$E[y(t)]$", fontsize=12)
    ax.set_title("Effect of fuzzy level $\\alpha$ on expected state trajectory",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.3, 1.1)

    fig.tight_layout()
    fig.savefig("fig/Fig10_Fuzzy_Level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig10 generated successfully")


if __name__ == "__main__":
    main()
