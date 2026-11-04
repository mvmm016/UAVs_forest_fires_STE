"""
experiments.py
--------------
Roda a simulação (UAV + filtro de partículas) várias vezes, em vários cenários,
e compara os resultados.

Uso no notebook:

    from experiments import run_experiments, summarize, plot_error_boxplots, plot_position_estimates
    results = run_experiments(scenarios, seeds=range(20))
    summarize(results)
    plot_error_boxplots(results)
    plot_position_estimates(results, scenarios)
"""
import io
import time
import contextlib
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from DispersionModels import gaussian_plume_gp
from Sensors import PIDGasSensor
from MovementStrategies import ParameterizedSweepStrategy
from SimulationClasses import World, UAV, DispersionModelConfig, MultiUAVHistory
from ParticleFilter import ParticleFilter, ParticleFilterConfig
from Likelihood import gaussian_likelihood
from Resample import systematic_resample


# ---------------------------------------------------------------------------
# Configuração padrão (mesmos valores da main). Cada cenário sobrescreve só o que muda.
# ---------------------------------------------------------------------------
DEFAULT_PRIORS = {
    "x_s":   {"dist": "uniform", "low": 0.0, "high": 40.0},
    "y_s":   {"dist": "uniform", "low": 0.0, "high": 40.0},
    "q":     {"dist": "gamma", "shape": 1.5, "scale": 1.0},
    "u_s":   {"dist": "normal", "mean": 1.0, "std": 0.2},
    "pitch": {"low": -np.pi / 2, "high": np.pi / 2},
    "yaw":   {"low": -np.pi, "high": np.pi},
    "zeta1": {"dist": "uniform", "low": 1.0, "high": 25.0},
    "zeta2": {"dist": "uniform", "low": 1.0, "high": 25.0},
}

DEFAULT_SCENARIO = {
    # --- Verdade (ground truth) ---
    "source_position": (10.0, 10.0, 1.5),
    "q": 1.0,
    "u": 5.0,
    "zetas": (0.3, 0.3),
    "wind_angles": (0.0, np.pi / 4),
    # --- UAV / trajetória ---
    "x_max": 40.0,
    "y_max": 40.0,
    "num_sectors": 12,
    "t_step": 0.005,
    # --- Sensores ---
    "mq4_noise_std": 0.001,
    "co2_noise_std": 0.005,
    # --- Filtro de partículas ---
    "num_particles": 3000,
    "n_eff_threshold": 0.5,
    "prior_params": DEFAULT_PRIORS,
    "pf_source_height": 0.0,  # altura da fonte assumida pelo filtro de partículas
}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _weighted_mean_std(values, weights):
    """Média e desvio padrão ponderados, ignorando partículas com NaN/inf."""
    ok = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not ok.any() or weights[ok].sum() <= 0:
        return np.nan, np.nan
    v = values[ok]
    w = weights[ok] / weights[ok].sum()
    mean = np.sum(w * v)
    std = np.sqrt(np.sum(w * (v - mean) ** 2))
    return float(mean), float(std)


# ---------------------------------------------------------------------------
# Uma simulação completa (sem plot em tempo real, sem prints)
# ---------------------------------------------------------------------------
def run_simulation(scenario: dict, seed: int = 0, keep_particles: bool = False) -> dict:
    sc = {**DEFAULT_SCENARIO, **scenario}
    np.random.seed(seed)  # todo o código usa np.random, então uma seed basta
    t_start = time.time()

    # 1. Mundo (verdade)
    env_config = DispersionModelConfig(
        source_position=sc["source_position"],
        q=sc["q"],
        u=sc["u"],
        zetas=sc["zetas"],
        wind_angles=sc["wind_angles"],
    )
    world = World(dispersion_model=gaussian_plume_gp, model_config=env_config)

    # 2. UAV e sensores
    strategy = ParameterizedSweepStrategy(
        sc["x_max"], sc["y_max"], sc["num_sectors"])
    uav = UAV("UAV_01", world, strategy)
    uav.add_sensor(PIDGasSensor("MQ4", min_resolution=1e-9,
                   noise_std=sc["mq4_noise_std"]))
    uav.add_sensor(PIDGasSensor(
        "CO2_Sensor", min_resolution=1e-3, noise_std=sc["co2_noise_std"]))

    t = 0.0
    world.register_uav(uav.id, strategy.compute_next_step(t))

    # 3. Estimador
    pf = ParticleFilter()
    pf_config = ParticleFilterConfig(
        num_particles=sc["num_particles"],
        n_eff_threshold=sc["n_eff_threshold"],
        resample_function=systematic_resample,
        likelihood_function=gaussian_likelihood,
        target_sensor_name="MQ4",
        dispersion_model=gaussian_plume_gp,
        source_height=sc["pf_source_height"],
    )
    particles = pf.initialize_particles(pf_config, sc["prior_params"])

    # 4. Loop da missão (o redirect_stdout silencia os print() do ParticleFilter.step)
    history = MultiUAVHistory()
    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            pos = uav.get_position()
            real_conc = world.get_real_concentration(pos)

            sensors_snapshot = {
                name: {"value": s.read_measurement(
                    real_conc), "noise_std": s.noise_std}
                for name, s in uav.sensors.items()
            }
            history.record_snapshot(
                {uav.id: {"position": pos, "sensors": sensors_snapshot}})

            particles = pf.step(
                history=history, memory=particles, config=pf_config)

            if t >= 1.0:
                break
            t = min(t + sc["t_step"], 1.0)
            uav.set_position(strategy.compute_next_step(t))

    # 5. Métricas
    x_est, x_std = _weighted_mean_std(particles["x_s"], particles["weight"])
    y_est, y_std = _weighted_mean_std(particles["y_s"], particles["weight"])
    q_est, q_std = _weighted_mean_std(particles["q"], particles["weight"])

    x_true, y_true, _ = sc["source_position"]
    q_true = sc["q"]
    mq4 = np.asarray(history.get_sensor_history(uav.id, "MQ4"))

    result = {
        "seed": seed,
        "x_est": x_est, "y_est": y_est, "q_est": q_est,
        "x_std": x_std, "y_std": y_std, "q_std": q_std,
        "pos_error": float(np.hypot(x_est - x_true, y_est - y_true)),
        "q_error": q_est - q_true,
        "q_rel_error": (q_est - q_true) / q_true,
        # medições MQ4 acima da resolução mínima (dentro da pluma)
        "n_hits": int(np.sum(mq4 > 0)),
        "n_measurements": int(len(mq4)),
        "runtime_s": time.time() - t_start,
    }
    if keep_particles:
        result["particles"] = particles.copy()
        result["trajectory"] = history.get_trajectory(uav.id)
        result["mq4"] = mq4.copy()
    return result


# ---------------------------------------------------------------------------
# Vários cenários x várias seeds
# ---------------------------------------------------------------------------
def run_experiments(scenarios: dict, seeds=range(20), keep_particles: bool = False, verbose: bool = True) -> dict:
    """
    Retorna {nome_do_cenario: [resultado_seed_0, resultado_seed_1, ...]}.
    Usa as MESMAS seeds em todos os cenários, para a comparação ser justa.
    """
    seeds = list(seeds)
    results = {}
    for name, sc in scenarios.items():
        runs = []
        for i, seed in enumerate(seeds, start=1):
            runs.append(run_simulation(sc, seed=seed,
                        keep_particles=keep_particles))
            if verbose:
                print(f"\r[{name}] {i}/{len(seeds)}", end="", flush=True)
        if verbose:
            print()
        results[name] = runs
    return results


def summarize(results: dict):
    """Tabela-resumo por cenário (impressa no terminal)."""
    header = (f"{'cenário':<18}{'erro pos (m)':>18}{'viés q':>10}{'RMSE q':>10}"
              f"{'erro rel q':>12}{'medições>0':>12}")
    print(header)
    print("-" * len(header))
    for name, runs in results.items():
        pos = np.array([r["pos_error"] for r in runs])
        qe = np.array([r["q_error"] for r in runs])
        qr = np.array([r["q_rel_error"] for r in runs])
        hits = np.array([r["n_hits"] for r in runs])
        print(f"{name:<18}"
              f"{np.nanmean(pos):>10.2f} ± {np.nanstd(pos):<5.2f}"
              f"{np.nanmean(qe):>10.3f}"
              f"{np.sqrt(np.nanmean(qe ** 2)):>10.3f}"
              f"{np.nanmean(qr) * 100:>11.1f}%"
              f"{hits.mean():>12.1f}")


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_error_boxplots(results: dict, save_path=None, show=True):
    """Boxplots do erro de posição e do erro relativo de q, um box por cenário."""
    names = list(results.keys())
    pos_data = [[r["pos_error"] for r in results[n]] for n in names]
    qrel_data = [[r["q_rel_error"] * 100 for r in results[n]] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, ylabel, title in (
        (axes[0], pos_data, "Erro de posição (m)", "Erro na posição da fonte"),
        (axes[1], qrel_data, "Erro relativo de q (%)", "Erro na taxa de emissão"),
    ):
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[1].axhline(0, color="crimson", linewidth=1.5, label="Sem erro")
    axes[1].legend()
    fig.suptitle(f"Resultados de {len(pos_data[0])} simulações por cenário")
    _finish(fig, save_path, show)


def plot_position_estimates(results: dict, scenarios: dict, save_path=None, show=True):
    """Um painel por cenário: posição estimada da fonte em cada seed vs. posição real."""
    names = list(results.keys())
    n = len(names)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(
        5 * cols, 4.5 * rows), squeeze=False)

    for ax, name in zip(axes.ravel(), names):
        sc = {**DEFAULT_SCENARIO, **scenarios[name]}
        xs = [r["x_est"] for r in results[name]]
        ys = [r["y_est"] for r in results[name]]
        ax.scatter(xs, ys, s=25, alpha=0.7,
                   color="royalblue", label="Estimativas")
        ax.scatter(sc["source_position"][0], sc["source_position"][1],
                   color="red", marker="X", s=140, edgecolors="black", label="Fonte real")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_title(name)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(
        "Posição estimada da fonte (uma nuvem de pontos = várias seeds)")
    _finish(fig, save_path, show)


def plot_q_posteriors(results: dict, scenarios: dict, bins=60, x_max_factor=4.0,
                      save_path=None, show=True):
    """
    Um painel por cenário. Cada linha fina azul é a posterior de q de UMA seed;
    a linha grossa é a média das posteriores. Requer keep_particles=True no run_experiments.
    """
    names = list(results.keys())
    n = len(names)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(
        6.5 * cols, 4.5 * rows), squeeze=False)

    for ax, name in zip(axes.ravel(), names):
        sc = {**DEFAULT_SCENARIO, **scenarios[name]}
        q_true = sc["q"]
        prior = sc["prior_params"]["q"]

        # Mesmos bins para todas as seeds, senão não dá para somar/comparar
        x_max = x_max_factor * q_true
        edges = np.linspace(0.0, x_max, bins + 1)
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0

        densities = []
        for i, r in enumerate(results[name]):
            p = r["particles"]
            q, w = p["q"], p["weight"]
            ok = np.isfinite(q) & np.isfinite(w) & (w >= 0)
            if not ok.any() or w[ok].sum() <= 0:
                continue
            hist, _ = np.histogram(
                q[ok], bins=edges, weights=w[ok] / w[ok].sum())
            dens = hist / widths
            densities.append(dens)
            ax.step(centers, dens, where="mid", color="royalblue", alpha=0.15, linewidth=1,
                    label="Seeds individuais" if i == 0 else None)

        ax.step(centers, np.mean(densities, axis=0), where="mid", color="navy",
                linewidth=2.5, label="Posterior média")

        # Priori gama
        grid = np.linspace(1e-6, x_max, 500)
        shape, scale = prior["shape"], prior["scale"]
        prior_pdf = grid ** (shape - 1) * np.exp(-grid /
                                                 scale) / (math.gamma(shape) * scale ** shape)
        ax.plot(grid, prior_pdf, "k--", linewidth=1.5, label="Priori")

        ax.axvline(q_true, color="crimson", linewidth=2, label="Verdade")
        ax.set_title(f"{name} ({len(densities)} seeds)")
        ax.set_xlabel(r"$q_s$ (g/s)")
        ax.set_ylabel("Densidade")
        ax.set_xlim(0, x_max)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle("Posterior da taxa de emissão q em várias simulações")
    _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = {
        "base":           dict(),  # usa só os valores padrão
        "vento_forte":    dict(u=10.0),
        "fonte_distante": dict(source_position=(30.0, 25.0, 1.5)),
        "q_alto":         dict(q=5.0),
    }

    results = run_experiments(scenarios, seeds=range(20))
    summarize(results)

    out = Path("resultados")
    plot_error_boxplots(results, save_path=out /
                        "erros_boxplot.png", show=False)
    plot_position_estimates(
        results, scenarios, save_path=out / "posicoes.png", show=False)
    print(f"Figuras salvas em {out.resolve()}")
