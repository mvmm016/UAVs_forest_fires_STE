from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

#Só define a interface: EstimatorConfig (base vazia) e SourceEstimator com o método abstrato step. Permite trocar o PF por outro estimador sem mudar o main.
@dataclass
class EstimatorConfig:
    """Base struct for estimator configurations."""
    pass

class SourceEstimator(ABC):
    @abstractmethod
    def step(self, uav_data: dict, memory: Any, config: EstimatorConfig) -> Any:
        """
        Executes one step of the estimation.
        Returns the updated memory (e.g., new particles).
        """
        pass