import torch.nn as nn
import torch

from utils.logger import logger
from utils.device_manager import get_device


class BaseModel(nn.Module):
    """
    An abstract base model class defining common functionalities for all models.
    
    This class can be extended to implement specific architectures or functionalities.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = get_device()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The forward method must be implemented by subclasses.")
    
    def to_device(self):
        self.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
