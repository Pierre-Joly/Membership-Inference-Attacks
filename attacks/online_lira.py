import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset
from datasets.subset import MembershipSubset
from utils.data_loader import get_data_loader
from utils.data_utils import get_shadow_dataset
from utils.device_manager import get_device
from utils.logger import logger
from utils.model_utils import clone_model
from utils.statistics import gaussian_pdf
from utils.train_utils import train_shadow_model

class OnlineLiRA(BaseAttack):
    def __init__(self, num_shadow: int = 64, batch_size: int = 128, reference_data: str = "data/pub.pt"):
        self.num_shadow = num_shadow
        self.batch_size = batch_size
        self.reference_data = reference_data

    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform Online Parallel Likelihood Ratio Attack (LiRA).

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Likelihood ratio scores for each data point.
        """
        device = get_device()
        model.to(device).eval()
        logger.info("Starting Online Parallel LiRA Attack")

        # Load and prepare public data
        combined_data = get_shadow_dataset(data, self.reference_data)

        # Train shadow models with index tracking
        shadow_models, inclusions = self.get_shadow_models(model, combined_data, self.num_shadow)

        # Precompute inclusion matrix (num_shadow x num_targets)
        num_targets = len(data)
        incl_matrix = self.create_inclusion_matrix(inclusions, num_targets)

        # Batch processing setup
        dataloader = get_data_loader(
            data, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        scores = np.zeros(len(data), dtype=np.float32)
        ptr = 0

        # Process batches
        for imgs, labels in tqdm(dataloader, desc="Processing Batches"):
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            # Get all shadow model confidences for this batch
            all_confs = []
            for shadow_model in shadow_models:
                with torch.no_grad():
                    outputs = shadow_model(imgs)
                    batch_confs = outputs[range(B), labels].cpu().numpy()
                    all_confs.append(batch_confs)
            all_confs = np.array(all_confs)  # Shape: (num_shadow, batch_size)

            # Get target model confidences
            with torch.no_grad():
                target_outputs = model(imgs)
                target_confs = target_outputs[range(B), labels].cpu().numpy()

            # Process the entire batch in parallel
            is_in_model = incl_matrix[:, ptr: ptr + B]  # Shape: (num_shadow, batch_size)
            confs_in = all_confs * is_in_model
            confs_out = all_confs * (~is_in_model)

            # Compute statistics
            eps = 1e-9
            mu_in = np.mean(confs_in, axis=0)
            sigma_in = np.std(confs_in, axis=0) + eps
            mu_out = np.mean(confs_out, axis=0)
            sigma_out = np.std(confs_out, axis=0) + eps

            # Compute likelihood ratios
            p_in = gaussian_pdf(target_confs, mu_in, sigma_in)
            p_out = gaussian_pdf(target_confs, mu_out, sigma_out)
            scores[ptr: ptr + B] = p_in / (p_out + eps)

            ptr += B

        logger.info("Online Parallel LiRA Attack Completed")
        return scores

    def get_shadow_models(self, base_model: nn.Module, combined_data: torch.utils.data.Dataset, num_models: int) -> tuple:
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
