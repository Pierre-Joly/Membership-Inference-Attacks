from torch.utils.data import Dataset, Subset
import torch
from typing import Tuple
from torchvision import transforms

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)
    
class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        self.membership = []
        mean = [0.2980, 0.2962, 0.2987]
        std = [0.2886, 0.2875, 0.2889]
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=mean, std=std)])
        super().__init__(transform)

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

class MembershipSubset(Subset):
    def __init__(self, dataset: Dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

def membership_collate_fn(batch):
    """
    Custom collation function to handle additional attributes.
    Args:
        batch: List of samples returned by __getitem__.
    Returns:
        Collated batch.
    """
    _, imgs, labels, _ = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    return imgs, labels
