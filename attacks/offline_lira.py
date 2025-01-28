import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset
from datasets.subset import MembershipSubset
from utils.data_loader import get_data_loader
from utils.data_utils import get_out_dataset
from utils.device_manager import get_device
from utils.logger import logger
from utils.model_utils import clone_model
from utils.statistics import gaussian_cdf
from utils.train_utils import train_shadow_model


class OfflineLiRA(BaseAttack):
    def __init__(self, num_shadow_models: int = 64, batch_size: int = 128, reference_data: str = "data/pub.pt"):
        self.num_shadow_models = num_shadow_models
        self.batch_size = batch_size
        self.reference_data = reference_data

    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform Offline Parallel Likelihood Ratio Attack (LiRA).

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Likelihood ratio scores for each data point.
        """
        device = get_device()
        model.to(device).eval()
        logger.info("Starting Offline Parallel LiRA Attack")

        # Get out of distribution data
        data_out = get_out_dataset(self.reference_data)

        # Define the shadow models
        shadow_models, inclusions = self.get_shadow_models(model, data_out, self.num_shadow_models)

        # Precompute inclusion matrix (num_shadow x num_targets)
        num_targets = len(data)
        incl_matrix = self.create_inclusion_matrix(inclusions, num_targets)

        # Prepare DataLoader for the target data
        data_loader = get_data_loader(
            data,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Initialize attack result array
        attack_results = np.zeros(len(data), dtype=np.float32)
        ptr = 0

        # Iterate over each example
        for imgs, labels in tqdm(data_loader, desc="Offline LiRA"):
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            confsout = torch.zeros(self.num_shadow_models, B, device=device)

            # Collect statistics for the current example
            for idx, shadow_model in enumerate(shadow_models):
                with torch.no_grad():
                    out_logits = shadow_model(imgs)  # shape [B, num_classes]
                    # Collect correct class logits
                    correct_logits = out_logits[range(B), labels]
                confsout[idx] = correct_logits

            # Compute Gaussian parameters for the 'out' distribution
            EPS = 1e-9
            mu_out = confsout.mean(dim=0)                 # shape [B]
            sigma_out = confsout.std(dim=0) + EPS         # shape [B]

            # Compute the observed logit from the target model
            with torch.no_grad():
                target_out = model(imgs)                         # [B, num_classes]
                conf_obs = target_out[range(B), labels]          # [B]

            # Compute the CDF Out
            cdf_out = gaussian_cdf(conf_obs, mu_out, sigma_out)

            attack_results[ptr: ptr+B] = cdf_out.cpu().numpy()
            ptr += B

        logger.info("Offline Parallel LiRA Attack Completed")
        return attack_results

    def get_shadow_models(self, base_model: nn.Module, data_out: MembershipDataset, num_models: int) -> tuple:
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
        inclusion_tracker = []  # Tracks which examples each model was trained on
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
        incl_matrix = np.zeros((self.num_shadow_models, num_targets), dtype=bool)
        for model_idx, indices in enumerate(inclusions):
            private_indices = [idx for idx in indices if idx < num_targets]
            incl_matrix[model_idx, private_indices] = True
        return incl_matrix
