from torch.utils.data import Dataset
import torch
from typing import Tuple, Optional, Any, List
from torchvision import transforms

from utils.logger import logger


class TaskDataset(Dataset):
    """
    A base dataset class for tasks involving image data and labels.

    This class provides a framework for datasets containing image data, unique identifiers, and labels. 
    It supports optional transformations to be applied to the images.

    Attributes:
        ids (List[int]): List of unique identifiers for each data point.
        imgs (List[torch.Tensor]): List of images as tensors.
        labels (List[int]): List of labels corresponding to each image.
        transform (Optional[torch.nn.Module]): Optional transformations to be applied to the images.
    """

    def __init__(self, transform: Optional[Any] = None):
        self.ids: List[int] = []
        self.imgs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return id_, img, label

    def __len__(self) -> int:
        return len(self.ids)


class MembershipDataset(TaskDataset):
    """
    A dataset class for membership inference attacks, extending TaskDataset.

    This class includes additional information indicating whether each data point is a member 
    of the training set (1) or not (0). It applies normalization transformations to the images.

    Attributes:
        membership (List[int]): List indicating membership status for each data point.
    """

    def __init__(self, data_path: str, transform: Optional[Any] = None):
        self.membership: List[int] = []

        mean = [0.2980, 0.2962, 0.2987]
        std = [0.2886, 0.2875, 0.2889]
        self.transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])

        super().__init__(self.transform)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        membership = self.membership[index]
        return id_, img, label, membership
