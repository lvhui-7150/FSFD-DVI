"""Reproducible numerical experiments for the FSFD-DVI paper revision.

The solver follows the discrete scheme in the manuscript:

    y^{n+1} = y^n - mem_n
            + Gamma(2-p) h^p (A y^n + phi^n)
            + Gamma(2-p) h^{p-1/2} sigma^n xi_n,

with mem_n = sum_{j=1}^n b_j (y^{n+1-j} - y^{n-j}).

For the corrected pathwise-coupled convergence study with confidence intervals,
use convergence_reproduction.py in this folder.

The fuzzy variational inequality is realized by the crisp singleton fuzzy
number

    [H_{(y, z, k)}]_{beta} = { k + 0.5 y - gamma z },

where beta in (0,1] is the constant alpha-level and gamma is the delayed-state
feedback coefficient displayed in the sensitivity study.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma


@dataclass
class Model:
    T: float = 3.0
    N: int = 500
    p: float = 0.75
    tau: float = 0.3
    gamma: float = 0.5
    sigma: float = 0.1
    init_val: float = 1.0
    level: float = 0.9
    use_fractional: bool = True
    use_delay: bool = True
    use_vi: bool = True
    fixed_control: float = 0.0

    @property
    def dt(self) -> float:
        return self.T / self.N

    def with_changes(self, **kwargs) -> "Model":
        for key in kwargs:
            if not hasattr(self, key):
                raise TypeError(f"unknown model field: {key}")
        return Model(**{**self.__dict__, **kwargs})


class FSFDSolver:
    def __init__(self, model: Model):
        self.model = model
        self.dt = model.dt

        if model.use_fractional:
            self.cp = gamma(2.0 - model.p)
            self.c_det = self.cp * self.dt ** model.p
            self.c_noise = self.cp * self.dt ** (model.p - 0.5)
            j = np.arange(model.N + 2)
            self.b = (j + 1) ** (1.0 - model.p) - j ** (1.0 - model.p)
        else:
            self.cp = 1.0
            self.c_det = self.dt
            self.c_noise = np.sqrt(self.dt)
            self.b = None

        tau = model.tau if model.use_delay else 0.0
        if tau > 0:
            n_hist = int(np.ceil(tau / self.dt))
            self.t_hist = np.linspace(-tau - self.dt, -self.dt, n_hist + 1)
        else:
            self.t_hist = np.array([-self.dt, -self.dt / 2.0])
        self.t_main = np.linspace(0.0, model.T, model.N + 1)
        self.t_all = np.concatenate([self.t_hist, self.t_main])
        self.start = len(self.t_hist)
        self.t = self.t_main

    def _delay_value(self, t_curr, t_grid, y_hist):
        target = t_curr - self.model.tau
        if target <= t_grid[0]:
            return y_hist[0]
        idx = int(np.searchsorted(t_grid, target, side="right")) - 1
        idx = max(0, min(idx, len(t_grid) - 2))
        t0, t1 = t_grid[idx], t_grid[idx + 1]
        y0, y1 = y_hist[idx], y_hist[idx + 1]
        return y0 + (y1 - y0) * (target - t0) / (t1 - t0)

    def solve_control(self, y, y_tau, k_prev):
        model = self.model
        if not model.use_vi:
            return model.fixed_control
        k = k_prev
        rho = 0.05
        for _ in range(40):
            grad = k + 0.5 * y - model.gamma * y_tau
            k_new = np.clip(k - rho * grad, -1.2, 1.2)
            if abs(k_new - k) < 1e-6:
                break
            k = k_new
        return k

    def simulate(self, dw=None):
        model = self.model
        if dw is None:
            rng = np.random.default_rng(0)
            dw = rng.normal(0.0, np.sqrt(self.dt), model.N)
        else:
            dw = np.asarray(dw, dtype=float)
            if len(dw) != model.N:
                raise ValueError("dw length must equal N")

        y = np.zeros(len(self.t_all))
        k = np.zeros(len(self.t_all))
        y[self.t_all <= 0.0] = model.init_val
        k[self.t_all <= 0.0] = model.fixed_control

        for n in range(self.start, len(self.t_all) - 1):
            t_n = self.t_all[n]
            if model.use_delay and model.tau > 0:
                y_tau = self._delay_value(
                    t_n, self.t_all[: n + 1], y[: n + 1]
                )
            else:
                y_tau = 0.0

            k[n] = self.solve_control(y[n], y_tau, k[n - 1])

            m = n - self.start
            if model.use_fractional and m >= 1:
                memory = float(
                    np.dot(
                        self.b[1 : m + 1][::-1],
                        np.diff(y[self.start : n + 1]),
                    )
                )
            else:
                memory = 0.0

            drift = -1.8 * y[n]
            if model.use_delay and model.tau > 0:
                drift += 0.3 * y_tau
            if model.use_vi:
                drift += k[n]

            xi = dw[n - self.start] / np.sqrt(self.dt)
            y[n + 1] = (
                y[n]
                - memory
                + self.c_det * drift
                + self.c_noise * model.sigma * y[n] * xi
            )

        return self.t_main.copy(), y[self.start :].copy(), k[self.start :].copy()


def mc_mean(solver: FSFDSolver, runs: int, rng: np.random.Generator):
    ys = []
    for _ in range(runs):
        dw = rng.normal(0.0, np.sqrt(solver.dt), solver.model.N)
        _, y, _ = solver.simulate(dw=dw)
        ys.append(y)
    arr = np.asarray(ys)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def mc_mean_for_models(models, runs, seed=1000):
    """Each model receives the same Wiener paths."""
    rng = np.random.default_rng(seed)
    ref_N = models[0].N
    ref_T = models[0].T
    dt = ref_T / ref_N
    dws = [rng.normal(0.0, np.sqrt(dt), ref_N) for _ in range(runs)]
    means = []
    for model in models:
        solver = FSFDSolver(model)
        if (model.N, model.T) != (ref_N, ref_T):
            raise ValueError("models must share N and T for shared paths")
        ys = [solver.simulate(dw=dw)[1] for dw in dws]
        means.append(np.mean(np.asarray(ys), axis=0))
    return np.asarray(means), dws


# Legacy helper retained for quick checks. The 200-path result reported in the
# paper is generated by convergence_reproduction.py.
def run_convergence(T=0.75, p=0.75, gamma=0.5, tau=0.3,
                    N_ref=8000, ns=(20, 40, 80, 160, 320),
                    runs=15, seed=999):
    ref_model = Model(
        T=T, N=N_ref, p=p, gamma=gamma, tau=tau, sigma=0.1,
        init_val=1.0, use_fractional=True, use_delay=True, use_vi=True,
    )
    ref_solver = FSFDSolver(ref_model)
    dt_ref = ref_model.dt

    rng = np.random.default_rng(seed)
    errors = {n: [] for n in ns}

    for _ in range(runs):
        z = rng.standard_normal(N_ref)
        dw_ref = np.sqrt(dt_ref) * z
        _, y_ref, _ = ref_solver.simulate(dw=dw_ref)
        for n in ns:
            if N_ref % n != 0:
                raise ValueError(f"{N_ref} must be divisible by {n}")
            q = N_ref // n
            model = Model(T=T, N=n, p=p, gamma=gamma, tau=tau, sigma=0.1,
                          init_val=1.0, use_fractional=True, use_delay=True,
                          use_vi=True)
            solver = FSFDSolver(model)
            # Sum q fine increments to get one coarse Brownian increment.
            dw_coarse = np.sqrt(dt_ref) * z.reshape(n, q).sum(axis=1)
            _, y_n, _ = solver.simulate(dw=dw_coarse)
            y_ref_matched = y_ref[::q][: len(y_n)]
            rmse = float(np.sqrt(np.mean((y_n - y_ref_matched) ** 2)))
            errors[n].append(rmse)

    hs = np.asarray([T / n for n in ns], dtype=float)
    means = np.asarray([float(np.mean(errors[n])) for n in ns])
    slope = float(np.polyfit(np.log(hs), np.log(means), 1)[0])
    return hs, means, slope, errors


def generate_all(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.dpi": 300,
    })

    # --- Fig 1-3 sensitivity -----------------------------------------
    print("Fig1-3: sensitivity runs ...", flush=True)
    configs = [
        ("p", [0.60, 0.75, 0.90], "Fig1_p_Sensitivity"),
        ("gamma", [0.0, 0.2, 0.5, 0.8], "Fig2_gamma_Sensitivity"),
        ("tau", [0.0, 0.3, 0.6], "Fig3_tau_Sensitivity"),
    ]
    for field, vals, fname in configs:
        models = []
        for val in vals:
            kwargs = {field: val, "T": 3.0, "N": 500, "p": 0.75,
                      "tau": 0.3, "gamma": 0.5, "sigma": 0.1}
            models.append(Model(**kwargs).with_changes(**{field: val}))
        means, _ = mc_mean_for_models(models, runs=50, seed=1000)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for val, mean in zip(vals, means):
            label = rf"{field}={val}"
            ax.plot(np.linspace(0, 3.0, len(mean)), mean,
                    lw=1.5, label=label)
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("E[y(t)]")
        ax.set_title(f"Sensitivity of {field}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, fname + ".png"))
        plt.close(fig)

    # --- Fig 4 stochastic stability ----------------------------------
    print("Fig4: stochastic stability ...", flush=True)
    model4 = Model(T=5.0, N=500, p=0.75, tau=0.3, gamma=0.5,
                   sigma=0.25, init_val=1.0)
    solver4 = FSFDSolver(model4)
    rng4 = np.random.default_rng(200)
    mean4, std4 = mc_mean(solver4, runs=150, rng=rng4)
    t4 = np.linspace(0, 5.0, len(mean4))
    fig4, ax4 = plt.subplots(figsize=(6.2, 4.2))
    ax4.plot(t4, mean4, "r-", lw=2, label="Expectation")
    ax4.fill_between(t4, mean4 - 1.96 * std4, mean4 + 1.96 * std4,
                     color="red", alpha=0.15, label="95% CI")
    ax4.set_xlabel("Time (t)")
    ax4.set_ylabel("E[y(t)]")
    ax4.set_title("Stochastic Stability")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "Fig4_Stochastic_Stability.png"))
    plt.close(fig4)

    # --- Fig 5 control variable --------------------------------------
    print("Fig5: control variable ...", flush=True)
    model5 = Model(T=5.0, N=500, p=0.75, tau=0.3, gamma=0.5,
                   sigma=0.1, init_val=1.0)
    solver5 = FSFDSolver(model5)
    rng5 = np.random.default_rng(7)
    _, _, k5 = solver5.simulate(dw=rng5.normal(0.0, np.sqrt(solver5.dt), model5.N))
    fig5, ax5 = plt.subplots(figsize=(6.2, 4.2))
    ax5.plot(np.linspace(0, 5.0, len(k5)), k5,
             color="forestgreen", lw=1)
    ax5.set_xlabel("Time (t)")
    ax5.set_ylabel("k(t)")
    ax5.set_title("Control Variable k(t)")
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "Fig5_Control_Variable.png"))
    plt.close(fig5)

    # --- Fig 6 convergence -------------------------------------------
    print("Fig6: convergence study ...", flush=True)
    hs, means, slope, _ = run_convergence()
    ns = [round(0.75 / h) for h in hs]
    fig6, ax6 = plt.subplots(figsize=(6.2, 4.2))
    ax6.loglog(ns, means, "D-", label="Measured RMSE", markersize=6)
    fit_line = means[0] * (ns[0] / np.asarray(ns)) ** slope
    ax6.loglog(ns, fit_line, "k--",
               label=rf"Fit slope $={slope:.3f}$")
    ax6.set_xlabel("Steps (N)")
    ax6.set_ylabel("RMSE")
    ax6.set_title("Convergence Rate Analysis")
    ax6.legend()
    ax6.grid(True, which="both", ls="-", alpha=0.2)
    fig6.tight_layout()
    fig6.savefig(os.path.join(output_dir, "Fig6_Convergence_Rate.png"))
    plt.close(fig6)
    with open(os.path.join(output_dir, "convergence_slope.txt"), "w") as fh:
        fh.write(f"slope={slope:.4f}\n")
        fh.write(f"runs=15\n")
        fh.write(f"N_ref=8000\n")

    # --- Fig 7 long-time behaviour ------------------------------------
    print("Fig7: long-time runs ...", flush=True)
    inits = [1.5, 0.8, 0.0, -0.6, -1.2]
    colors = ["#E63946", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
    fig7, ax7 = plt.subplots(figsize=(10.0, 5.0))
    for val, color in zip(inits, colors):
        model7 = Model(T=15.0, N=500, p=0.8, tau=0.3, gamma=0.5,
                       sigma=0.08, init_val=val)
        solver7 = FSFDSolver(model7)
        rng7 = np.random.default_rng(300)
        mean7, _ = mc_mean(solver7, runs=80, rng=rng7)
        t7 = np.linspace(0, 15.0, len(mean7))
        ax7.plot(t7, mean7, lw=2, label=rf"Initial $\chi(s)={val}$",
                 color=color)
    ax7.set_xlabel("Time (t)")
    ax7.set_ylabel("E[y(t)]")
    ax7.set_title("Long-time Evolution of Mean State")
    ax7.legend(ncol=3)
    ax7.grid(True, linestyle="--", alpha=0.5)
    fig7.tight_layout()
    fig7.savefig(os.path.join(output_dir, "Fig7_Global_Stability.png"))
    plt.close(fig7)

    # --- Fig 8 ablation ------------------------------------------------
    print("Fig8: ablation runs ...", flush=True)
    base8 = dict(T=6.0, N=600, p=0.8, tau=0.3, gamma=0.5,
                 sigma=0.1, init_val=1.0)
    variants = [
        ("Full model", dict(use_fractional=True, use_delay=True, use_vi=True)),
        ("Integer order", dict(use_fractional=False, use_delay=True, use_vi=True)),
        ("No delay", dict(use_fractional=True, use_delay=False, use_vi=True)),
        ("No fuzzy control", dict(use_fractional=True, use_delay=True, use_vi=False,
                                  fixed_control=0.0)),
        ("Pure SDE", dict(use_fractional=False, use_delay=False, use_vi=False)),
    ]
    models8 = [Model(**{**base8, **cfg}) for _, cfg in variants]
    means8, dws8 = mc_mean_for_models(models8, runs=80, seed=400)
    t8 = np.linspace(0, 6.0, len(means8[0]))
    colors8 = ["#1D3557", "#457B9D", "#E63946", "#F4A261", "#2A9D8F"]
    fig8, ax8 = plt.subplots(figsize=(10.0, 6.0))
    for (label, _), mean, color in zip(variants, means8, colors8):
        ax8.plot(t8, mean, lw=2, label=label, color=color)
    ax8.set_xlabel("Time (t)")
    ax8.set_ylabel("E[y(t)]")
    ax8.set_title("Ablation Study")
    ax8.legend(loc="best")
    ax8.grid(True, linestyle=":", alpha=0.6)
    fig8.tight_layout()
    fig8.savefig(os.path.join(output_dir, "Fig8_Ablation_Study.png"))
    plt.close(fig8)

    # --- Fig 9 full model vs pure SDE ----------------------------------
    print("Fig9: advanced comparison ...", flush=True)
    model_full = Model(T=6.0, N=600, p=0.8, tau=0.3, gamma=0.6,
                       sigma=0.15, init_val=1.0)
    model_simple = Model(T=6.0, N=600, p=0.8, tau=0.3, gamma=0.6,
                         sigma=0.15, init_val=1.0,
                         use_fractional=False, use_delay=False, use_vi=False)
    solver_full = FSFDSolver(model_full)
    solver_simple = FSFDSolver(model_simple)
    rng9 = np.random.default_rng(500)
    ys_full = []
    ys_simple = []
    for _ in range(100):
        dw = rng9.normal(0.0, np.sqrt(model_full.dt), model_full.N)
        ys_full.append(solver_full.simulate(dw=dw)[1])
        ys_simple.append(solver_simple.simulate(dw=dw)[1])
    arr_full = np.asarray(ys_full)
    arr_simple = np.asarray(ys_simple)
    t9 = np.linspace(0, 6.0, arr_full.shape[1])
    fig9, ax9 = plt.subplots(figsize=(10.0, 6.0))
    for path in arr_simple[:10]:
        ax9.plot(t9, path, color="gray", alpha=0.10, lw=0.8)
    ax9.plot(t9, np.mean(arr_simple, axis=0), color="#E63946",
             ls="--", lw=2, label="Simplified SDE")
    mean9 = np.mean(arr_full, axis=0)
    std9 = np.std(arr_full, axis=0)
    ax9.fill_between(t9, mean9 - 1.96 * std9, mean9 + 1.96 * std9,
                     color="#457B9D", alpha=0.2, label="95% CI")
    ax9.plot(t9, mean9, color="#1D3557", lw=3, label="FSFD-DVI model")
    ax9.set_xlabel("Time (t)")
    ax9.set_ylabel("y(t)")
    ax9.set_title("Full Model versus Pure SDE")
    ax9.legend(loc="upper right")
    ax9.grid(True, linestyle=":", alpha=0.6)
    fig9.tight_layout()
    fig9.savefig(os.path.join(output_dir, "Fig9_Advanced_Comparison.png"))
    plt.close(fig9)

    print("Saved figures to", os.path.abspath(output_dir))
    print(f"Convergence fit slope: {slope:.4f}")


if __name__ == "__main__":
    generate_all("figures_revised")
