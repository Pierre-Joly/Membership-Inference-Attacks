import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import ConcatDataset

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset, transform_test
from datasets.subset import MembershipSubset
from utils.data_loader import get_data_loader
from utils.data_utils import get_out_dataset
from utils.device_manager import get_device
from utils.logger import logger
from utils.statistics import compute_quantiles 
from utils.shadow_models import get_off_shadow_models


class OfflineRMIA(BaseAttack):
    def __init__(self, num_shadow_models: int = 64,
                batch_size: int = 128,
                reference_data: str = "data/pub.pt",
                population_size:int = 5000,
                a_param: float = 0.3,
                gamma: float = 1.0):
        self.num_shadow_models = num_shadow_models
        self.batch_size = batch_size
        self.reference_data = reference_data
        self.population_size = population_size
        self.a_param = a_param
        self.gamma = gamma

    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform Offline Robust Membership Inference Attack (RMIA).

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Likelihood ratio scores for each data point.
        """
        device = get_device()
        model.to(device).eval()
        logger.info("Starting Offline RMIA Attack")

        ###
        data_pub: MembershipDataset = torch.load(self.reference_data, map_location='cpu', weights_only=False)
        member_indices = [i for i, m in enumerate(data_pub.membership) if m == 1]
        non_member_indices = [i for i, m in enumerate(data_pub.membership) if m == 0]
        data_in = MembershipSubset(data_pub, member_indices)
        data_out = MembershipSubset(data_pub, non_member_indices)
        data = ConcatDataset([data_in, data_out])
        ###

        # Get out of distribution data
        data_out = get_out_dataset(self.reference_data)

        # Define the shadow models
        shadow_models = get_off_shadow_models(model, data_out, self.num_shadow_models)

        ###

        # Change transform for inference
        transform = transform_test()
        data_out.transform = transform
        data.transform = transform

        ###

        # Get loader of population z
        z_loader = self.get_z_loader(data_out, self.population_size, self.batch_size)

        # Compute population ratio
        population_ratio = np.zeros(self.population_size, dtype=np.float32)
        self.get_ratio(population_ratio, model, shadow_models, z_loader, device, "Computing z ratio")

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
        self.get_ratio(ratio_x, model, shadow_models, data_loader, device, "Computing x ratio")

        # Get attacks results
        attack_results = np.zeros(len(data), dtype=np.float32)
        scaled_ratio_x = ratio_x / self.gamma
        attack_results = compute_quantiles(population_ratio, scaled_ratio_x)

        logger.info("Offline RMIA Attack Completed")
        return attack_results
    
    
    def get_ratio(self, ratio, model, shadow_models, loader, device, description):
        ptr = 0

        with torch.no_grad():
            for imgs, labels in tqdm(loader, total=len(loader), desc=description):
                imgs, labels = imgs.to(device), labels.to(device)

                B = imgs.size(0)

                # Compute Pr(. | \theta)
                target = F.softmax(model(imgs), dim=1) # [B, num_classes]
                conf = target[range(B), labels].cpu().numpy() # [B]

                # Compute Pr(.)_{OUT}
                shadow_conf = np.zeros((self.num_shadow_models, B), dtype=np.float32) # [num_shadow_models, B]
                for idx, sm in enumerate(shadow_models):
                    probs_sm = F.softmax(sm(imgs), dim=1) # [B, num_classes]
                    conf_sm = probs_sm[range(B), labels].cpu().numpy() # [B]
                    shadow_conf[idx] = conf_sm # [num_shadow_models, B]

                pr_out = shadow_conf.mean(axis=0) # [B]

                # Apply linear correction
                pr = 0.5 * ((1.0 + self.a_param) * pr_out + (1.0 - self.a_param))

                # Compute Pr(. | \theta) / Pr(.)
                ratio[ptr: ptr+B] = conf / np.maximum(pr, 1e-12)

                ptr += B
    

    def get_z_loader(self, data_out, population_size, batch_size):
        z_indices = np.random.choice(len(data_out), size=self.population_size, replace=False)
        z_subset = MembershipSubset(data_out, z_indices)
        z_loader = get_data_loader(z_subset, batch_size=self.batch_size, shuffle=False)
        return z_loader
    
