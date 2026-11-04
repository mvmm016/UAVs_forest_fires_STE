import math

import numpy as np
import matplotlib.pyplot as plt


#plot_q_estimate: filtra partículas válidas, normaliza pesos, calcula média e desvio ponderados de q, e plota o histograma ponderado (posterior), a priori gama, média ± DP e o valor verdadeiro. Retorna (mean_q, std_q).
def plot_q_estimate(particles, true_q, prior_params, bins=80):
    """Plot the weighted posterior estimate and Gamma prior for q."""
    q_values = np.asarray(particles['q'], dtype=float)
    weights = np.asarray(particles['weight'], dtype=float)
    valid = np.isfinite(q_values) & np.isfinite(
        weights) & (q_values >= 0.0) & (weights >= 0.0)
    q_values = q_values[valid]
    weights = weights[valid]

    if q_values.size == 0 or np.sum(weights) <= 0.0:
        raise ValueError("No valid q particles are available for plotting.")

    weights = weights / np.sum(weights)
    mean_q = np.average(q_values, weights=weights)
    std_q = np.sqrt(np.average((q_values - mean_q) ** 2, weights=weights))

    q_max = max(float(np.max(q_values)), float(true_q), 1.0)
    q_grid = np.linspace(0.0, q_max * 1.15, 500)
    histogram, edges = np.histogram(
        q_values, bins=bins, range=(0.0, q_grid[-1]), weights=weights)
    widths = np.diff(edges)
    posterior_density = histogram / widths
    centers = edges[:-1] + widths / 2.0

    shape = float(prior_params["shape"])
    scale = float(prior_params["scale"])
    prior_density = np.zeros_like(q_grid)
    positive = q_grid > 0.0
    prior_density[positive] = (
        q_grid[positive] ** (shape - 1.0)
        * np.exp(-q_grid[positive] / scale)
        / (math.gamma(shape) * scale ** shape)
    )

    plt.figure(figsize=(9, 5.5))
    plt.step(centers, posterior_density, where="mid",
             color="royalblue", linewidth=2, label="Estimate PDF")
    plt.plot(q_grid, prior_density, "k--", linewidth=1.5, label="Prior")
    plt.axvline(mean_q, color="limegreen",
                linestyle=":", linewidth=2, label="Mean")
    plt.axvline(mean_q - std_q, color="limegreen",
                linestyle=":", linewidth=1.5)
    plt.axvline(mean_q + std_q, color="limegreen", linestyle=":",
                linewidth=1.5, label="Mean +/- SD")
    plt.axvline(true_q, color="crimson", linewidth=2, label="Truth")
    plt.xlabel(r"$q_s$ (g/s)")
    plt.ylabel("Density")
    plt.title("Source Emission Rate Estimate")
    plt.grid(True, linestyle=":", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_q, std_q


#plot_particle_filter_step: animação por iteração: limpa a figura, desenha a pluma real, a trajetória, o UAV atual, as partículas coloridas por peso (as de maior peso por cima) e a fonte real.
def plot_particle_filter_step(particles, trajectory, grid_x, grid_y, Z, source_pos, step, pause_time=0.05):
    """
    Visualiza a distribuição das partículas em 2D sobreposta à pluma real e à trajetória.
    """
    plt.clf()  # Limpa o frame anterior para criar a animação

    # Background Plume (Ground Truth)
    if Z is not None:
        plt.contourf(grid_x, grid_y, Z, levels=50, cmap='jet', alpha=0.3)

    # Trajetória do UAV
    if trajectory:
        traj_pts = np.array(trajectory)
        plt.plot(traj_pts[:, 0], traj_pts[:, 1],
                 'w--', alpha=0.8, label='Trajetória')
        # Destaca a posição atual do drone
        plt.scatter(traj_pts[-1, 0], traj_pts[-1, 1], color='cyan',
                    marker='^', s=150, edgecolors='black', label='UAV')

    # Partículas
    # Ordenamos pelas partículas de maior peso para que sejam desenhadas por cima
    order = np.argsort(particles['weight'])
    ordered_particles = particles[order]

    scatter = plt.scatter(
        ordered_particles['x_s'], ordered_particles['y_s'],
        c=ordered_particles['weight'], cmap='viridis',
        s=15, alpha=0.8, label='Partículas'
    )
    plt.colorbar(scatter, label='Peso da Partícula')

    # Posição Real da Fonte
    plt.scatter(source_pos[0], source_pos[1], color='red',
                marker='X', s=150, label='Fonte Real')

    plt.title(f"Filtro de Partículas - Iteração {step}")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Atualiza a tela e devolve o controle para o loop principal
    plt.pause(pause_time)


#plot_mission_results: figura final com pluma, trajetória, medições (tamanho do círculo proporcional à leitura), fonte e seta do vento.
def plot_mission_results(grid_x, grid_y, Z, traj_history, meas_history, source_pos, wind_angles):
    plt.figure(figsize=(10, 8))

    # Background Plume
    heatmap = plt.contourf(grid_x, grid_y, Z, levels=100,
                           cmap='jet', alpha=0.5)
    plt.colorbar(heatmap, label=r'Concentration $(g/m^3)$')

    # Trajectory
    traj_pts = np.array(traj_history)
    plt.plot(traj_pts[:, 0], traj_pts[:, 1],
             'w--', alpha=0.7, label='UAV Path')

    # Measurements
    sizes = np.array(meas_history) * 5000 + 10
    plt.scatter(traj_pts[:, 0], traj_pts[:, 1], s=sizes, c='white',
                edgecolors='black', alpha=0.8, label='Measurements')

    # Conversão dos Ângulos para Projeção do Vetor 2D (Plano XY)
    pitch, yaw = wind_angles
    u_x = np.cos(yaw) * np.cos(pitch)
    u_y = np.sin(yaw) * np.cos(pitch)

    # Source and Wind
    plt.scatter(source_pos[0], source_pos[1], color='red',
                marker='X', s=100, label='Source')
    # Usando u_x e u_y para desenhar a seta de direção
    plt.quiver(source_pos[0], source_pos[1], u_x, u_y,
               color='black', scale=10, label='Wind Direction')

    plt.title("UAV Mission Simulation")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
