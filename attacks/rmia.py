import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from attacks.base_attack import BaseAttack
from datasets.dataset import MembershipDataset
from utils.data_loader import get_data_loader
from utils.data_utils import get_out_dataset
from utils.device_manager import get_device
from utils.logger import logger
from utils.statistics import compute_quantiles

class RMIA(BaseAttack):
    def __init__(self, metric: str = "loss", reference_data: str = "data/pub.pt", batch_size: int = 128):
        self.metric = metric
        self.reference_data = reference_data
        self.batch_size = batch_size

    def run_attack(self, model: nn.Module, data: MembershipDataset) -> np.ndarray:
        """
        Perform Robust Membership Inference Attack (RMIA).

        Args:
            model (nn.Module): The target model.
            data (MembershipDataset): The dataset to attack.

        Returns:
            np.ndarray: Membership scores for each data point.
        """
        device = get_device()
        model.to(device).eval()
        logger.info("Starting RMIA Attack")

        # Get out of distribution data
        data_out = get_out_dataset(self.reference_data)

        # Prepare DataLoader for the public dataset
        data_loader_out = get_data_loader(
            data_out,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Iterate over public dataset
        reference_metrics = np.zeros(len(data_out), dtype=np.float32)
        ptr = 0

        for imgs, labels in tqdm(data_loader_out, desc="RMIA Public"):
            with torch.no_grad():
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                B = imgs.size(0)

                if self.metric == "loss":
                    batch_metrics = F.cross_entropy(outputs, labels, reduction='none')
                elif self.metric == "confidence":
                    probs = F.softmax(outputs, dim=1)
                    batch_metrics = probs[torch.arange(B), labels]
                elif self.metric == "entropy":
                    probs = F.softmax(outputs, dim=1)
                    batch_metrics = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
                else:
                    raise ValueError(f"Unknown metric '{self.metric}'. Choose from 'loss', 'confidence', or 'entropy'.")

                reference_metrics[ptr:ptr+B] = batch_metrics.cpu().numpy()
                ptr += B

        # Sorting reference metrics
        reference_metrics.sort()

        # Prepare DataLoader for the target dataset
        data_loader_target = get_data_loader(
            data,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Iterate over target dataset
        membership_scores = np.zeros(len(data), dtype=np.float32)
        ptr = 0

        for imgs, labels in tqdm(data_loader_target, desc="RMIA Target"):
            with torch.no_grad():
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                B = imgs.size(0)

                if self.metric == "loss":
                    batch_metrics = F.cross_entropy(outputs, labels, reduction='none')
                elif self.metric == "confidence":
                    probs = F.softmax(outputs, dim=1)
                    batch_metrics = probs[torch.arange(B), labels]
                elif self.metric == "entropy":
                    probs = F.softmax(outputs, dim=1)
                    batch_metrics = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
                else:
                    raise ValueError(f"Unknown metric '{self.metric}'. Choose from 'loss', 'confidence', or 'entropy'.")

                # Convert to numpy array
                batch_metrics_np = batch_metrics.cpu().numpy()

                # Compute quantile scores for the batch
                batch_scores = compute_quantiles(reference_metrics, batch_metrics_np)

                # Invert scores for loss and entropy
                if self.metric in ["loss", "entropy"]:
                    batch_scores = 1 - batch_scores

                membership_scores[ptr:ptr+B] = batch_scores
                ptr += B

        logger.info("RMIA Attack Completed")
        return membership_scores
