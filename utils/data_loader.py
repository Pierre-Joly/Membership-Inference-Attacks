from torch.utils.data import DataLoader
from typing import Any

from datasets.dataset import MembershipDataset
from utils.logger import logger

import torch
from typing import List, Tuple

def membership_collate_fn(batch: List[Tuple[int, torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, imgs, labels, _ = zip(*batch)

    # Convert to tensors
    imgs = torch.stack(imgs)  # Stack image tensors
    labels = torch.tensor(labels, dtype=torch.int64)  # Labels

    return imgs, labels


def get_data_loader(dataset: MembershipDataset, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (MembershipDataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        DataLoader: Configured DataLoader.
    """
    logger.info(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=membership_collate_fn,
        pin_memory=True,
        num_workers=4
    )
