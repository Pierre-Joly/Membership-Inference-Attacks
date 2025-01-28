import copy
import torch.nn as nn

from utils.device_manager import get_device
from utils.logger import logger

def clone_model(base_model: nn.Module) -> nn.Module:
    """
    Clone the base model. If no base model is provided, initialize a new ResNet18.

    Args:
        base_model (nn.Module, optional): The base model to clone. Defaults to None.

    Returns:
        nn.Module: Cloned model.
    """
    cloned_model = copy.deepcopy(base_model)
    cloned_model.to(get_device())
    logger.info("Cloned model moved to device")
    return cloned_model
