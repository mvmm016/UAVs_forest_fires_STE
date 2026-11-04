import numpy as np
from abc import ABC, abstractmethod


class Sensor(ABC):
    def __init__(self, name: str, min_resolution: float):
        self.name = name
        self.min_resolution = min_resolution

#Sensor (classe abstrata): exige noise_std (propriedade) e read_measurement().
    @property
    @abstractmethod
    def noise_std(self) -> float:
        """
        Retorna o desvio padrão do ruído do sensor.
        Contrato obrigatório para que os estimadores (ex: Filtro de Partículas) 
        possam acessar esse valor e calcular a verossimilhança.
        """
        pass

    @abstractmethod
    def read_measurement(self, real_concentration: float) -> float:
        """
        Applies sensor-specific noise and resolution limits to the ground-truth value.
        """
        pass

#PIDGasSensor: soma ruído gaussiano N(0, noise_std) à concentração real. Se o resultado ficar abaixo de min_resolution, devolve 0.0, o que simula o limite de detecção.
class PIDGasSensor(Sensor):
    def __init__(self, name: str, min_resolution: float, noise_std: float = 0.001):
        super().__init__(name, min_resolution)
        # Salva o valor em uma variável privada (com o underscore na frente)
        self._noise_std = noise_std

    @property
    def noise_std(self) -> float:
        # Implementa o getter exigido pela classe abstrata
        return self._noise_std

    def read_measurement(self, real_concentration: float) -> float:
        # Usa a property para pegar o ruído
        noise = np.random.normal(0, self.noise_std)
        meas = real_concentration + noise

        # Applies minimum resolution threshold
        if meas < self.min_resolution:
            return 0.0
        return float(meas)