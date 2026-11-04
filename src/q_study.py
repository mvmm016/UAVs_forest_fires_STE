"""
q_study.py
----------
Estudo: "Como o aumento de Q impacta o desempenho do filtro de partículas?"

Colocar em src/, ao lado de experiments.py. Uso no notebook:

    from q_study import run_q_study, summarize_q, plot_q_sweep
    all_results, all_scenarios = run_q_study(seeds=range(50))
    summarize_q(all_results, all_scenarios)
    plot_q_sweep(all_results, all_scenarios, lang="pt", save_path="resultados/varredura_Q.png")

Por convenção do projeto, as estimativas pontuais usam a MEDIANA PONDERADA
(a média/desvio não são estimativas válidas sob posteriores multimodais).
"""
import numpy as np
import matplotlib.pyplot as plt

from experiments import run_experiments, DEFAULT_SCENARIO, DEFAULT_PRIORS, _finish
from DispersionModels import gaussian_plume_gp


# ---------------------------------------------------------------------------
# Configuração do estudo
# ---------------------------------------------------------------------------
Q_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Priori de Q larga (média 10): testa se o viés vem da priori padrão, que quase não tem massa em Q alto
PRIORS_Q_LARGA = {**DEFAULT_PRIORS, "q": {"dist": "gamma", "shape": 2.0, "scale": 5.0}}

# Prioris de zeta e u_s que contêm os valores verdadeiros: testa o efeito do modelo mal especificado
PRIORS_FISICA_OK = {
    **DEFAULT_PRIORS,
    "zeta1": {"dist": "uniform", "low": 0.1, "high": 1.0},
    "zeta2": {"dist": "uniform", "low": 0.1, "high": 1.0},
    "u_s":   {"dist": "normal", "mean": 5.0, "std": 1.0},
}

PRIOR_CONFIGS = {
    "priori padrão": None,
    "priori de Q larga": PRIORS_Q_LARGA,
    "ζ e u corrigidos": PRIORS_FISICA_OK,
}


def make_sweeps(q_values=Q_VALUES, prior_configs=PRIOR_CONFIGS):
    """{rótulo da configuração: {"Q=0.5": cenário, "Q=1": cenário, ...}}"""
    sweeps = {}
    for label, priors in prior_configs.items():
        sweeps[label] = {
            f"Q={q:g}": ({"q": q} if priors is None else {"q": q, "prior_params": priors})
            for q in q_values
        }
    return sweeps


def run_q_study(seeds=range(50), q_values=Q_VALUES, prior_configs=PRIOR_CONFIGS):
    """Roda todas as varreduras. Retorna (resultados, cenários), ambos indexados pelo rótulo da configuração."""
    sweeps = make_sweeps(q_values, prior_configs)
    all_results = {}
    for label, scenarios in sweeps.items():
        print(f"== {label} ==")
        all_results[label] = run_experiments(scenarios, seeds=seeds, keep_particles=True)
    return all_results, sweeps


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def weighted_median(values, weights):
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cum, 0.5)])


def run_metrics(run, scenario):
    """Erro de posição e razão Q_est/Q_verdadeiro, usando a mediana ponderada das partículas."""
    sc = {**DEFAULT_SCENARIO, **scenario}
    p = run["particles"]
    x = weighted_median(p["x_s"], p["weight"])
    y = weighted_median(p["y_s"], p["weight"])
    q = weighted_median(p["q"], p["weight"])
    return {
        "pos_error": float(np.hypot(x - sc["source_position"][0], y - sc["source_position"][1])),
        "q_ratio": q / sc["q"],
    }


def detectable_fraction(run, scenario, n_sigma=3.0):
    """
    Fração da trajetória em que a concentração REAL (sem ruído) passa de n_sigma vezes o ruído do sensor.
    É uma medida de quanta informação a missão pode extrair da pluma. Não depende da seed.
    """
    sc = {**DEFAULT_SCENARIO, **scenario}
    conc = np.array([
        gaussian_plume_gp(pos, sc["source_position"], sc["q"], sc["u"], sc["zetas"], sc["wind_angles"])
        for pos in run["trajectory"]
    ], dtype=float)
    return float(np.mean(conc > n_sigma * sc["mq4_noise_std"]))


def collect(results, scenarios, success_radius=5.0):
    """Uma linha de métricas por valor de Q."""
    rows = []
    for name, runs in results.items():
        sc = scenarios[name]
        m = [run_metrics(r, sc) for r in runs]
        pos = np.array([a["pos_error"] for a in m])
        ratio = np.array([a["q_ratio"] for a in m])
        rows.append({
            "name": name,
            "Q": {**DEFAULT_SCENARIO, **sc}["q"],
            "pos_med": np.nanmedian(pos),
            "pos_q25": np.nanpercentile(pos, 25), "pos_q75": np.nanpercentile(pos, 75),
            "ratio_med": np.nanmedian(ratio),
            "ratio_q25": np.nanpercentile(ratio, 25), "ratio_q75": np.nanpercentile(ratio, 75),
            "success": float(np.mean(pos < success_radius)),
            "detect": detectable_fraction(runs[0], sc),
        })
    return rows


def summarize_q(all_results, all_scenarios, success_radius=5.0):
    for label in all_results:
        rows = collect(all_results[label], all_scenarios[label], success_radius)
        print(f"\n== {label} ==")
        header = (f"{'Q':>6}{'erro pos mediano (m)':>24}{'Q_est/Q mediano':>18}"
                  f"{'sucesso (<' + f'{success_radius:g}' + ' m)':>16}{'traj. detectável':>18}")
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['Q']:>6g}"
                  f"{r['pos_med']:>14.2f} [{r['pos_q25']:.1f}–{r['pos_q75']:.1f}]"
                  f"{r['ratio_med']:>12.2f}"
                  f"{r['success'] * 100:>15.0f}%"
                  f"{r['detect'] * 100:>17.0f}%")


# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------
LABELS = {
    "pt": {
        "x": "Q verdadeiro (g/s)",
        "ratio": "Q estimado / Q verdadeiro", "ratio_t": "Viés na taxa de emissão",
        "pos": "Erro de posição (m)", "pos_t": "Erro na posição da fonte",
        "succ": "Seeds com erro < {r:g} m (%)", "succ_t": "Taxa de sucesso",
        "det": "Trajetória com pluma > 3σ (%)", "det_t": "Detectabilidade da pluma",
        "truth": "Sem erro", "title": "Efeito de Q no filtro de partículas (mediana e intervalo interquartil)",
    },
    "en": {
        "x": "True Q (g/s)",
        "ratio": "Estimated Q / true Q", "ratio_t": "Emission-rate bias",
        "pos": "Position error (m)", "pos_t": "Source position error",
        "succ": "Seeds with error < {r:g} m (%)", "succ_t": "Success rate",
        "det": "Path with plume > 3σ (%)", "det_t": "Plume detectability",
        "truth": "No error", "title": "Effect of Q on the particle filter (median and interquartile range)",
    },
}


def plot_q_sweep(all_results, all_scenarios, lang="pt", success_radius=5.0, save_path=None, show=True):
    L = LABELS[lang]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (ax_r, ax_p), (ax_s, ax_d) = axes

    for label in all_results:
        rows = collect(all_results[label], all_scenarios[label], success_radius)
        Q = np.array([r["Q"] for r in rows])

        line, = ax_r.plot(Q, [r["ratio_med"] for r in rows], "o-", label=label)
        ax_r.fill_between(Q, [r["ratio_q25"] for r in rows], [r["ratio_q75"] for r in rows],
                          color=line.get_color(), alpha=0.15)

        line, = ax_p.plot(Q, [r["pos_med"] for r in rows], "o-", label=label)
        ax_p.fill_between(Q, [r["pos_q25"] for r in rows], [r["pos_q75"] for r in rows],
                          color=line.get_color(), alpha=0.15)

        ax_s.plot(Q, [r["success"] * 100 for r in rows], "o-", label=label)
        # A detectabilidade só depende de Q (a priori não entra), então uma curva basta
        ax_d.plot(Q, [r["detect"] * 100 for r in rows], "o-", color="black")

    ax_r.axhline(1.0, color="crimson", linewidth=1.5, label=L["truth"])
    ax_r.set_yscale("log")

    for ax, ylabel, title in ((ax_r, L["ratio"], L["ratio_t"]), (ax_p, L["pos"], L["pos_t"]),
                              (ax_s, L["succ"].format(r=success_radius), L["succ_t"]),
                              (ax_d, L["det"], L["det_t"])):
        ax.set_xscale("log")
        ax.set_xlabel(L["x"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)

    ax_r.legend(fontsize=8)
    ax_s.set_ylim(-5, 105)
    ax_d.set_ylim(-5, 105)
    fig.suptitle(L["title"])
    _finish(fig, save_path, show)