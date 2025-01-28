from models.base import BaseModel
from torchvision import models
import torch.nn as nn
import torch


class ResNet18Model(BaseModel):
    """
    ResNet18 model tailored for the Membership Inference Attack framework.
    
    Extends the BaseModel to implement a specific architecture.
    """
    
    def __init__(self, num_classes: int = 44):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.to_device()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = "model." + key
            new_state_dict[new_key] = value
        
        super(ResNet18Model, self).load_state_dict(new_state_dict, strict=strict)
