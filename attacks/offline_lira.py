import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset
from utils.data_loader import get_data_loader
from utils.data_utils import get_out_dataset
from utils.device_manager import get_device
from utils.logger import logger
from utils.statistics import gaussian_cdf
from utils.shadow_models import get_off_shadow_models


class OfflineLiRA(BaseAttack):
    def __init__(self, num_shadow_models: int = 64, batch_size: int = 128, reference_data: str = "data/pub.pt"):
        self.num_shadow_models = num_shadow_models
        self.batch_size = batch_size
        self.reference_data = reference_data

    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform Offline Likelihood Ratio Attack (LiRA).

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Likelihood ratio scores for each data point.
        """
        device = get_device()
        model.to(device).eval()
        logger.info("Starting Offline LiRA Attack")

        # Get out of distribution data
        data_out = get_out_dataset(self.reference_data)

        # Define the shadow models
        shadow_models = get_off_shadow_models(data_out, self.num_shadow_models)

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
                    out_logits = shadow_model(imgs)  # [B, num_classes]
                    # Collect correct class logits
                    correct_logits = out_logits[range(B), labels]
                confsout[idx] = correct_logits

            # Compute Gaussian parameters for the 'out' distribution
            EPS = 1e-9
            mu_out = confsout.mean(dim=0)                 # [B]
            sigma_out = confsout.std(dim=0) + EPS         # [B]

            # Compute the observed logit from the target model
            with torch.no_grad():
                target_out = model(imgs)                         # [B, num_classes]
                conf_obs = target_out[range(B), labels]          # [B]

            # Compute the CDF Out
            cdf_out = gaussian_cdf(conf_obs, mu_out, sigma_out)

            attack_results[ptr: ptr+B] = cdf_out.cpu().numpy()
            ptr += B

        logger.info("Offline LiRA Attack Completed")
        return attack_results

