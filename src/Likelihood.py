import numpy as np

#Calcula a PDF normal de cada predição em torno da medição, com variância noise_std². Some 1e-300 para evitar peso exatamente zero.
def gaussian_likelihood(predictions: np.ndarray, measurement: float, noise_std: float = 0.5) -> np.ndarray:
    """
    Calcula a verossimilhança de cada partícula usando uma distribuição normal.
    Agnóstica ao modelo físico, depende apenas dos arrays de predição e da leitura real.
    
    :param predictions: Array (N,) com as predições geradas pelo modelo de dispersão.
    :param measurement: O valor escalar real lido pelo sensor do UAV.
    :param noise_std: Desvio padrão do ruído do sensor.
    :return: Array (N,) com as verossimilhanças.
    """
    variance = noise_std ** 2
    
    # Função densidade de probabilidade (PDF) da Gaussiana
    likelihoods = (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((predictions - measurement) ** 2) / (2 * variance))
    
    # Adiciona um valor ínfimo para evitar pesos matematicamente zerados (underflow)
    likelihoods += 1e-300
    
    return likelihoods