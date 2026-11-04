import numpy as np

import numpy as np

def systematic_resample(particles: np.ndarray, measurements: np.ndarray = None) -> np.ndarray:
    """
    Implementação da Reamostragem Sistemática (O(N)).
    Minimiza a variância da amostragem em comparação com o método multinomial.
    Aplica um kernel Gaussiano (roughening) no final para manter a diversidade.
    """
    N = len(particles)
    weights = particles['weight']

    #Soma cumulativa dos pesos.
    # Array cumulativo de pesos
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0 # Garante que termina exatamente em 1

    #Um único ponto de partida aleatório em [0, 1/N) e N pontos igualmente espaçados a partir dele.
    # Cria o passo e o ponto de partida inicial aleatório
    step = 1.0 / N
    random_start = np.random.uniform(0, step)
    
    # Array de pontos de amostragem
    points = random_start + np.arange(N) * step

    #searchsorted mapeia cada ponto a uma partícula. Quem tem mais peso é duplicada mais vezes e quem tem pouco peso some.
    # Encontra os índices usando busca binária
    indices = np.searchsorted(cumulative_sum, points)
    
    # Cria um novo array de partículas reamostradas
    resampled_particles = particles[indices].copy()
    
    # ----------------------------------------------------------------
    # --- APLICAÇÃO DO KERNEL GAUSSIANO (Roughening / Random Walk) ---
    # ----------------------------------------------------------------
    
    #Roughening: adiciona ruído gaussiano em cada parâmetro para as cópias não ficarem idênticas (mantém diversidade).
    # Desvios padrões da perturbação para cada variável. 
    # (Ajuste esses valores de acordo com a escala do seu ambiente/modelo)
    noise_stds = {
        'x_s': 0.5,    # Posição X (ex: meio metro de ruído)
        'y_s': 0.5,    # Posição Y 
        'q': 0.1,      # Taxa de emissão
        'u_s': 0.2,    # Velocidade do vento
        'pitch': 0.05, # Ângulos em radianos
        'yaw': 0.05,
        'zeta1': 0.2,  # Coeficientes de difusão
        'zeta2': 0.2
    }
    
    for key, std in noise_stds.items():
        # Adiciona o ruído normal N(0, std) a todas as partículas para esta variável
        resampled_particles[key] += np.random.normal(0, std, N)
        
    # --- Segurança Física (Limitações) ---
    # O ruído aleatório pode gerar valores absurdos para a física do modelo.
    # Clipamos (limitamos) os valores para garantir que façam sentido.
    
    # A taxa de emissão (q) e os coeficientes de difusão não podem ser <= 0
    #resampled_particles['q'] = np.clip(resampled_particles['q'], 1e-3, None)
    #resampled_particles['zeta1'] = np.clip(resampled_particles['zeta1'], 1e-3, None)
    #resampled_particles['zeta2'] = np.clip(resampled_particles['zeta2'], 1e-3, None)
    
    # Opcional: Manter o Pitch entre -pi/2 e pi/2 e o Yaw entre -pi e pi
    #resampled_particles['pitch'] = np.clip(resampled_particles['pitch'], -np.pi/2, np.pi/2)
    # O Yaw, se passar de pi, é melhor "dar a volta" (wrap) do que clipar:
    #resampled_particles['yaw'] = (resampled_particles['yaw'] + np.pi) % (2 * np.pi) - np.pi
    
    return resampled_particles