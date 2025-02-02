import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from datasets.dataset import MembershipDataset
from datasets.subset import MembershipSubset
from utils.model_utils import clone_model
from utils.train_utils import train_shadow_model


def get_off_shadow_models(self, base_model: nn.Module, data_out: MembershipDataset, num_models: int) -> tuple:
    """
    Train shadow models on out-of-distribution data.

    Args:
        base_model (nn.Module): The base model to clone.
        data_out (MembershipDataset): Out-of-distribution data.
        num_models (int): Number of shadow models to train.

    Returns:
        tuple: Shadow models and their inclusion indices.
    """
    shadow_models = []
    sample_size = len(data_out) // 2

    for _ in tqdm(range(num_models), desc="Training Shadow Models"):
        # Sample indices from data_out
        indices = np.random.choice(len(data_out), size=sample_size, replace=False).tolist()
        subset_out = MembershipSubset(data_out, indices)

        # Clone and train model
        model_clone = clone_model(base_model)
        train_shadow_model(model_clone, subset_out)
        model_clone.eval()

        shadow_models.append(model_clone)

    return shadow_models


def get_on_shadow_models(self, base_model: nn.Module, combined_data: torch.utils.data.Dataset, num_models: int) -> tuple:
    """
    Train shadow models and track their inclusions.

    Args:
        base_model (nn.Module): The base model to clone.
        combined_data (torch.utils.data.Dataset): Combined shadow dataset.
        num_models (int): Number of shadow models to train.

    Returns:
        tuple: Shadow models and their inclusion indices.
    """
    shadow_models = []
    inclusion_tracker = []  # Tracks which examples each model was trained on
    sample_size = len(combined_data) // 2

    for _ in tqdm(range(num_models), desc="Training Shadow Models"):
        # Sample indices from combined dataset
        indices = np.random.choice(len(combined_data), size=sample_size, replace=False).tolist()
        subset = MembershipSubset(combined_data, indices)

        # Clone and train model
        model_clone = clone_model(base_model)
        train_shadow_model(model_clone, subset)
        model_clone.eval()

        shadow_models.append(model_clone)
        inclusion_tracker.append(set(indices))

    return shadow_models, inclusion_tracker


def create_inclusion_matrix(self, inclusions: list, num_targets: int) -> np.ndarray:
    """
    Create an inclusion matrix indicating which shadow models include which targets.

    Args:
        inclusions (list): Inclusion indices for each shadow model.
        num_targets (int): Number of target data points.

    Returns:
        np.ndarray: Inclusion matrix of shape (num_shadow, num_targets).
    """
    incl_matrix = np.zeros((self.num_shadow, num_targets), dtype=bool)
    for model_idx, indices in enumerate(inclusions):
        private_indices = [idx for idx in indices if idx < num_targets]
        incl_matrix[model_idx, private_indices] = True
    return incl_matrix
    
