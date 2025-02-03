import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from datasets.subset import MembershipSubset
from utils.model_utils import clone_model
from utils.logger import logger
from utils.train_utils import single_worker, ddp_worker, train_shadow_model_common
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
    sample_size = len(data_out) // 2  # Size of the training subset for each model

    for _ in tqdm(range(num_models), desc="Training Shadow Models"):
        # Select a random subset for training this shadow model
        indices = np.random.choice(len(data_out), size=sample_size, replace=False).tolist()
        subset_out = MembershipSubset(data_out, indices)
        
        state_dict = single_worker(subset_out, batch_size=256)
        
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
    sample_size = len(combined_data) // 2

    for _ in tqdm(range(num_models), desc="Training Shadow Models"):
        # Sample indices from combined dataset
        indices = np.random.choice(len(combined_data), size=sample_size, replace=False).tolist()
        in_indices = [idx for idx in indices if getattr(combined_data, 'membership', [0]*len(combined_data))[idx] == 1]

        subset = MembershipSubset(combined_data, indices)

        # Clone and train model
        device = get_device()
        model_clone = clone_model(device=device)
        train_shadow_model_common(model=model_clone, subset=subset, device=device)
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
    
