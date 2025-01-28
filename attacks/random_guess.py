import numpy as np
import torch.nn as nn

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset
from utils.logger import logger

class RandomGuessAttack(BaseAttack):
    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform a random guess attack.

        Args:
            model (nn.Module): The target model (unused).
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Random membership scores.
        """
        logger.info("Executing Random Guess Attack")
        return np.random.rand(len(data))
