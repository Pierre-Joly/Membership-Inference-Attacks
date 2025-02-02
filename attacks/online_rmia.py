import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset
from datasets.subset import MembershipSubset
from utils.data_loader import get_data_loader
from utils.data_utils import get_shadow_dataset
from utils.device_manager import get_device
from utils.logger import logger
from utils.statistics import compute_quantiles 
from utils.shadow_models import get_on_shadow_models, create_inclusion_matrix


class OnlineRMIA(BaseAttack):
    def __init__(self, num_shadow_models: int = 64,
                batch_size: int = 128,
                reference_data: str = "data/pub.pt",
                population_size:int = 5000,
                gamma: float = 1.0):
        self.num_shadow_models = num_shadow_models
        self.batch_size = batch_size
        self.reference_data = reference_data
        self.population_size = population_size
        self.gamma = gamma

    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform Online Robust Membership Inference Attack (RMIA).

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Likelihood ratio scores for each data point.
        """
        device = get_device()
        model.to(device).eval()
        logger.info("Starting Online RMIA Attack")

        # Load and prepare public data
        combined_data = get_shadow_dataset(data, self.reference_data)

        # Train shadow models with index tracking
        shadow_models, inclusions = get_on_shadow_models(combined_data, self.num_shadow_models)

        # Precompute inclusion matrix (num_shadow x num_targets)
        num_targets = len(data)
        incl_matrix = create_inclusion_matrix(self.num_shadow_models, inclusions, num_targets)

        # Get loader of population z
        z_loader = self.get_z_loader(combined_data, self.population_size, self.batch_size)

        # Compute population ratio
        population_ratio = np.zeros(self.population_size, dtype=np.float32)
        self.get_ratio(population_ratio, model, shadow_models, z_loader, incl_matrix, device, "Computing z ratio")

        # Sort to efficiently compute quantiles
        population_ratio.sort()

        # Prepare DataLoader for the target data
        data_loader = get_data_loader(
            data,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Initialize attack result array
        ratio_x = np.zeros(len(data), dtype=np.float32)
        self.get_ratio(ratio_x, model, shadow_models, data_loader, incl_matrix, device, "Computing x ratio")

        # Get attacks results
        attack_results = np.zeros(len(data), dtype=np.float32)
        scaled_ratio_x = ratio_x / self.gamma
        attack_results = compute_quantiles(population_ratio, scaled_ratio_x)

        logger.info("Online RMIA Attack Completed")
        return attack_results
    

    def get_ratio(self, ratio, model, shadow_models, loader, incl_matrix, device, description):
        ptr = 0

        with torch.no_grad():
            for (imgs, labels) in tqdm(loader, desc=description):
                imgs, labels = imgs.to(device), labels.to(device)

                B = imgs.size(0)

                # Compute Pr(. | \theta)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                conf = probs[range(B), labels].cpu().numpy()  # shape [B]

                # Compute Pr(.)_{OUT}
                shadow_conf = np.zeros((self.num_shadow_models, B), dtype=np.float32) # [num_shadow_models, B]
                for idx, sm in enumerate(shadow_models):
                    probs_sm = F.softmax(sm(imgs), dim=1) # [B, num_classes]
                    conf_sm = probs_sm[range(B), labels].cpu().numpy() # [B]
                    shadow_conf[idx] = conf_sm # [num_shadow_models, B]

                # Compute Pr(.)_{IN} and Pr(z)_{OUT}
                pr = np.zeros(B, dtype=np.float32)

                is_in_model = incl_matrix[:, ptr: ptr + B]  # [num_shadow, batch_size]
                confs_in = shadow_conf * is_in_model
                confs_out = shadow_conf * (~is_in_model)

                pr_in = confs_in.mean(axis=0)
                pr_out = confs_out.mean(axis=0)

                # Compute Pr(z)
                pr = 0.5 * (pr_in + pr_out)

                # Compute Pr(z | \theta) / Pr(z)
                ratio[ptr: ptr + B] = conf / np.maximum(pr, 1e-12)

                ptr += B


    def get_z_loader(self, data, population_size, batch_size):
        z_indices = np.random.choice(len(data), size=self.population_size, replace=False)
        z_subset = MembershipSubset(data, z_indices)
        z_loader = get_data_loader(z_subset, batch_size=self.batch_size, shuffle=False)
        return z_loader
    
