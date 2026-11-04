import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
from Estimators import SourceEstimator, EstimatorConfig

# ParticleFilterConfig: número de partículas, limiar de N_eff, função de reamostragem, nome do sensor usado, modelo de dispersão e função de verossimilhança (tudo injetado, por isso é modular).


@dataclass
class ParticleFilterConfig(EstimatorConfig):
    num_particles: int
    n_eff_threshold: float
    resample_function: Callable
    target_sensor_name: str
    dispersion_model: Callable
    likelihood_function: Callable
    source_height: float = 0.0

# __init__: define o dtype estruturado da partícula: peso + os parâmetros a estimar (x_s, y_s, q, u_s, roll, pitch, yaw, zeta1, zeta2).


class ParticleFilter(SourceEstimator):
    def __init__(self):
        # Array estruturado das partículas
        self.particle_dtype = np.dtype([
            ('weight', np.float64),
            ('x_s', np.float64),
            ('y_s', np.float64),
            ('q', np.float64),
            ('u_s', np.float64),
            ('roll', np.float64),
            ('pitch', np.float64),
            ('yaw', np.float64),
            ('zeta1', np.float64),
            ('zeta2', np.float64)
        ])

# initialize_particles: cria N partículas com peso 1/N e sorteia cada parâmetro da sua priori (uniforme para posição, ângulos e zetas; gama para q; normal para u_s).
    def initialize_particles(self, config: ParticleFilterConfig, prior_params: dict) -> np.ndarray:
        """
        Inicializa o array estruturado de partículas amostrando das distribuições a priori.
        """
        N = config.num_particles

        particles = np.zeros(N, dtype=self.particle_dtype)

        particles['weight'] = 1.0 / N

        particles['x_s'] = np.random.uniform(
            prior_params["x_s"]["low"], prior_params["x_s"]["high"], N)
        particles['y_s'] = np.random.uniform(
            prior_params["y_s"]["low"], prior_params["y_s"]["high"], N)
        particles['q'] = np.random.gamma(
            prior_params["q"]["shape"], prior_params["q"]["scale"], N)
        particles['u_s'] = np.random.normal(
            prior_params["u_s"]["mean"], prior_params["u_s"]["std"], N)
        particles['pitch'] = np.random.uniform(
            prior_params["pitch"]["low"], prior_params["pitch"]["high"], N)
        particles['yaw'] = np.random.uniform(
            prior_params["yaw"]["low"], prior_params["yaw"]["high"], N)
        particles['zeta1'] = np.random.uniform(
            prior_params["zeta1"]["low"], prior_params["zeta1"]["high"], N)
        particles['zeta2'] = np.random.uniform(
            prior_params["zeta2"]["low"], prior_params["zeta2"]["high"], N)

        return particles

    def step(self, history: Any, memory: Any, config: EstimatorConfig) -> np.ndarray:
        """
        Executa um passo de estimação.
        """
        particles = memory

        # Array para acumular as verossimilhanças de todos os drones no instante atual
        # Começa com combined_likelihoods = 1.
        combined_likelihoods = np.ones(len(particles), dtype=np.float64)

        # Pega a lista de todos os VANTs registrados no objeto de histórico
        uav_ids = history.get_all_uav_ids()

        for uav_id in uav_ids:
            # Recupera as listas completas deste drone
            trajectory = history.get_trajectory(uav_id)
            sensor_data = history.get_sensor_history(
                uav_id, config.target_sensor_name)

            # Se não houver dados, pula para o próximo drone
            if not trajectory or not sensor_data:
                continue

            # Extrai os dados do instante atual (o mais recente é sempre o último elemento)
            x_k, y_k, z_k = trajectory[-1]
            measurement = sensor_data[-1]
            z_s = config.source_height

            # Recupera o desvio padrão do sensor específico deste UAV
            # Para cada UAV: pega a última posição e a última medição, e o noise_std do sensor.
            noise_std = history.get_sensor_noise_std(
                uav_id, config.target_sensor_name)

            # TODO: Predição (movimento aleatório das partículas / random walk)???
            # Investigar melhor a necessidade disso....talvez modularizar....

            # --- Atualização (Modelo de Dispersão) ----
            # Predição da medição: roda o modelo de dispersão com os parâmetros de cada partícula, obtendo a concentração que cada uma "esperaria" ali.
            c_preds = config.dispersion_model(
                sensor_pos=(x_k, y_k, z_k),
                source_pos=(particles['x_s'], particles['y_s'], z_s),
                qs=particles['q'],
                us=particles['u_s'],
                zetas=(particles['zeta1'], particles['zeta2']),
                wind_euler=(particles['pitch'], particles['yaw'])
            )

            # Verossimilhança: compara predição × medição real e multiplica no acumulado (fusão de UAVs independentes).
            # --- Verossimilhança ---
            likelihoods = config.likelihood_function(
                c_preds, measurement, noise_std=noise_std)

            # Multiplica a verossimilhança (fusão de dados dos múltiplos sensores independentes)
            combined_likelihoods *= likelihoods

        # Atualização: weight *= combined_likelihoods, e então normaliza (com proteção se tudo zerar).
        # Aplica a verossimilhança final agregada aos pesos das partículas
        particles['weight'] *= combined_likelihoods

        # --- Normalização dos pesos ---
        weight_sum = np.sum(particles['weight'])
        if weight_sum < 1e-300:  # Previne divisão por zero se todas as partículas morrerem
            particles['weight'] = 1.0 / len(particles)
        else:
            particles['weight'] /= weight_sum

        # N_eff: 1/Σw² dividido por N, ou seja, a fração de partículas "efetivas". Se cair abaixo do limiar, há degeneração.
        # --- Reamostragem ---
        n_eff = (1.0 / np.sum(particles['weight'] ** 2)) / config.num_particles
        print(n_eff)
        if n_eff < config.n_eff_threshold:
            print("Reamostrou!")
            # 1. Extrai todo o histórico de medições do target_sensor de todos os VANTs
            all_measurements = []
            uav_ids = history.get_all_uav_ids()
            for uav_id in uav_ids:
                sensor_history = history.get_sensor_history(
                    uav_id, config.target_sensor_name)
                if sensor_history:
                    all_measurements.extend(sensor_history)

            # 2. Converte para um array NumPy
            measurements_array = np.array(all_measurements)

            # Reamostragem: se necessário, chama a função de reamostragem e volta os pesos para 1/N.
            # 3. Chama a função de reamostragem com a nova assinatura
            particles = config.resample_function(particles, measurements_array)

            # 4. Restaura os pesos para uma distribuição uniforme
            particles['weight'] = 1.0 / len(particles)

        return particles
