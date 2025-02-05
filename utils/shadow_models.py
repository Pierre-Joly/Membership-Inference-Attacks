import torch
import numpy as np
from tqdm import tqdm

from datasets.subset import MembershipSubset
from utils.model_utils import clone_model
from utils.train_utils import single_worker, train_shadow_model_common
from utils.device_manager import get_device


def get_off_shadow_models(data_out, num_models: int) -> tuple:
    """
    Train several shadow models on out-of-distribution data, adapting the training mode
    (multi-GPU DDP or single-device) based on the available hardware.
    
    Args:
        data_out (MembershipDataset): Out-of-distribution data.
        num_models (int): Number of shadow models to train.
    
    Returns:
        tuple: A tuple of trained shadow models.
    """
    shadow_models = []
    split_idx = len(data_out) // 2
    lenght = len(data_out)

    for _ in tqdm(range(num_models), desc="Training Shadow Models"):
        # Select a random subset for training this shadow model
        indices = np.random.choice(lenght, size=lenght, replace=False).tolist()

        indices_train, indices_val = indices[:split_idx], indices[split_idx:]

        subset_train = MembershipSubset(data_out, indices_train)
        subset_val = MembershipSubset(data_out, indices_val)
        
        state_dict = single_worker(subset_train, subset_val, batch_size=256)
        
        # Clone the base model and load the trained state_dict
        device = get_device()
        model_clone = clone_model(device=device)
        model_clone.load_state_dict(state_dict, strict=True)
        model_clone.eval()
        shadow_models.append(model_clone)
    
    return tuple(shadow_models)


def get_on_shadow_models(combined_data: torch.utils.data.Dataset, num_models: int) -> tuple:
    """
    Train shadow models and track their inclusions.

    Args:
        combined_data (torch.utils.data.Dataset): Combined shadow dataset.
        num_models (int): Number of shadow models to train.

    Returns:
        tuple: Shadow models and their inclusion indices.
    """
    shadow_models = []
    inclusion_tracker = []  # Tracks which examples each model was trained on
    split_idx = len(combined_data) // 2
    lenght = len(combined_data)

    for _ in tqdm(range(num_models), desc="Training Shadow Models"):
        # Sample indices from combined dataset
        indices = np.random.choice(lenght, size=lenght, replace=False).tolist()

        indices_train, indices_val = indices[:split_idx], indices[split_idx:]

        in_indices = [idx for idx in indices_train if getattr(combined_data, 'membership', [0]*len(combined_data))[idx] == 1]

        subset_train = MembershipSubset(combined_data, indices_train)
        subset_val = MembershipSubset(combined_data, indices_val)

        state_dict = single_worker(subset_train, subset_val, batch_size=256)

        # Clone the base model and load the trained state_dict
        device = get_device()
        model_clone = clone_model(device=device)
        model_clone.load_state_dict(state_dict, strict=True)
        model_clone.eval()
        shadow_models.append(model_clone)

        inclusion_tracker.append(set(in_indices))

    return shadow_models, inclusion_tracker


def create_inclusion_matrix(num_shadow: int, inclusions: list, num_targets: int) -> np.ndarray:
    """
    Create an inclusion matrix indicating which shadow models include which targets.

    Args:
        inclusions (list): Inclusion indices for each shadow model.
        num_targets (int): Number of target data points.

    Returns:
        np.ndarray: Inclusion matrix of shape (num_shadow, num_targets).
    """
    incl_matrix = np.zeros((num_shadow, num_targets), dtype=bool)
    for model_idx, indices in enumerate(inclusions):
        incl_matrix[model_idx, list(indices)] = True
    return incl_matrix
    
