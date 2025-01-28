from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch.nn as nn

class BaseAttack(ABC):
    @abstractmethod
    def run_attack(self, model: nn.Module, data: Any) -> np.ndarray:
        """
        Execute the membership inference attack.

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Membership scores.
        """
        pass
