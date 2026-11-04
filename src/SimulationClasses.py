from dataclasses import dataclass
from typing import Callable, Tuple
from MovementStrategies import MovementStrategy
from Sensors import Sensor

@dataclass
class DispersionModelConfig:
    """Armazena os parâmetros reais do ambiente (Ground Truth)."""
    source_position: Tuple[float, float, float]
    q: float
    u: float
    zetas: Tuple[float, float]
    wind_angles: Tuple[float, float]

class World:
    def __init__(self, dispersion_model: Callable, model_config: DispersionModelConfig):
        self.uav_positions = {}
        self.dispersion_model = dispersion_model
        self.model_config = model_config

    def register_uav(self, uav_id: str, initial_position: tuple):
        self.uav_positions[uav_id] = initial_position

    def get_real_position(self, uav_id: str):
        return self.uav_positions.get(uav_id)

    def set_real_position(self, uav_id: str, new_pos: tuple):
        self.uav_positions[uav_id] = new_pos

    def get_real_concentration(self, pos: tuple):
        """
        O World chama a função do modelo passando as configurações armazenadas.
        Se a configuração for alterada em tempo de execução, a concentração muda dinamicamente.
        """
        return self.dispersion_model(
            pos,
            self.model_config.source_position,
            self.model_config.q,
            self.model_config.u,
            self.model_config.zetas,
            self.model_config.wind_angles
        )

class UAV:
    def __init__(self, uav_id: str, world: World, strategy: MovementStrategy):
        self.id = uav_id
        self.world = world
        self.strategy = strategy
        self.sensors = {} # Dictionary to hold multiple sensors by name

    def add_sensor(self, sensor: Sensor):
        self.sensors[sensor.name] = sensor

    def get_sensor(self, name: str) -> Sensor:
        if name not in self.sensors:
            raise ValueError(f"Sensor '{name}' not found on UAV '{self.id}'")
        return self.sensors[name]

    def get_position(self):
        return self.world.get_real_position(self.id)

    def set_position(self, new_pos):
        self.world.set_real_position(self.id, new_pos)

from typing import Dict, List, Tuple

class MultiUAVHistory:
    def __init__(self):
        # Estrutura interna: { "uav_id": { "trajectory": [...], "sensors": { "MQ4": [...], ... } } }
        self._data: Dict[str, Dict] = {}

    def record_snapshot(self, uav_data_snapshot: dict):
        """Registra a posição e as leituras de sensores de múltiplos UAVs no instante atual."""
        for uav_id, data in uav_data_snapshot.items():
            if uav_id not in self._data:
                # Inicializa a estrutura para um novo UAV
                self._data[uav_id] = {
                    "trajectory": [],
                    "sensors": {},
                    "sensor_metadata": {} # Novo: Guarda informações estáticas
                }
            
            # Salva a posição
            self._data[uav_id]["trajectory"].append(data["position"])
            
            for sensor_name, sensor_info in data["sensors"].items():
                # Se é a primeira vez que vemos este sensor, salvamos os metadados
                if sensor_name not in self._data[uav_id]["sensors"]:
                    self._data[uav_id]["sensors"][sensor_name] = []
                    self._data[uav_id]["sensor_metadata"][sensor_name] = {
                        "noise_std": sensor_info["noise_std"]
                    }
                
                # Armazena apenas a leitura na lista de histórico
                self._data[uav_id]["sensors"][sensor_name].append(sensor_info["value"])

    def get_trajectory(self, uav_id: str) -> List[Tuple[float, float, float]]:
        """Retorna o histórico de posições de um UAV específico."""
        return self._data.get(uav_id, {}).get("trajectory", [])

    def get_sensor_history(self, uav_id: str, sensor_name: str) -> List[float]:
        """Retorna o histórico de medições de um sensor específico de um UAV."""
        return self._data.get(uav_id, {}).get("sensors", {}).get(sensor_name, [])

    def get_sensor_noise_std(self, uav_id: str, sensor_name: str) -> float:
        """Recupera o desvio padrão do sensor armazenado nos metadados."""
        return self._data.get(uav_id, {}).get("sensor_metadata", {}).get(sensor_name, {}).get("noise_std", 1.0)
    
    def get_all_uav_ids(self) -> List[str]:
        """Retorna a lista de todos os UAVs rastreados."""
        return list(self._data.keys())