import torch.nn as nn

from models.resnet18 import ResNet18Model
from utils.logger import logger

def clone_model(device) -> nn.Module:
    """
    Clone the base model. If no base model is provided, initialize a new ResNet18.

    Args:
        base_model (nn.Module, optional): The base model to clone. Defaults to None.

    Returns:
        nn.Module: Cloned model.
    """
    cloned_model = ResNet18Model()
    cloned_model.to(device)
    logger.info("Cloned model moved to device")
    return cloned_model

