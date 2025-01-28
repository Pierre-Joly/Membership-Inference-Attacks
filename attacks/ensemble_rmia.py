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

class Ensemble_RMIA(BaseAttack):
    def __init__(self, reference_data: str = "data/pub.pt", batch_size: int = 128):
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
        logger.info("Starting Ensemble RMIA Attack")

        # Get out of distribution data
        data_out = get_out_dataset(self.reference_data)

        # Prepare DataLoader for the public dataset
        data_loader_out = get_data_loader(
            data_out,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Iterate over public dataset
        reference_loss = np.zeros(len(data_out), dtype=np.float32)
        reference_confidence = np.zeros(len(data_out), dtype=np.float32)
        reference_entropy = np.zeros(len(data_out), dtype=np.float32)
        ptr = 0

        for imgs, labels in tqdm(data_loader_out, desc="Ensemble RMIA Public"):
            with torch.no_grad():
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                B = imgs.size(0)

                # Loss
                batch_loss = F.cross_entropy(outputs, labels, reduction='none')

                # Confidence
                probs = F.softmax(outputs, dim=1)
                batch_confidence = probs[torch.arange(B), labels]

                # Entropy
                probs = F.softmax(outputs, dim=1)
                batch_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)

                # Assign metrics to reference arrays
                reference_loss[ptr:ptr+B] = batch_loss.cpu().numpy()
                reference_confidence[ptr:ptr+B] = batch_confidence.cpu().numpy()
                reference_entropy[ptr:ptr+B] = batch_entropy.cpu().numpy()
                ptr += B

        # Sorting reference metrics
        reference_loss.sort()
        reference_confidence.sort()
        reference_entropy.sort()

        # Prepare DataLoader for the target dataset
        data_loader_target = get_data_loader(
            data,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Iterate over target dataset
        membership_scores = np.zeros(len(data), dtype=np.float32)
        ptr = 0

        for imgs, labels in tqdm(data_loader_target, desc="Ensemble RMIA Target"):
            with torch.no_grad():
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                B = imgs.size(0)

                # Loss
                batch_loss = F.cross_entropy(outputs, labels, reduction='none').cpu().numpy()

                # Confidence
                probs = F.softmax(outputs, dim=1)
                batch_confidence = probs[torch.arange(B), labels].cpu().numpy()

                # Entropy
                probs = F.softmax(outputs, dim=1)
                batch_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).cpu().numpy()

                # Compute quantile scores for the batch
                batch_scores_loss = 1 - compute_quantiles(reference_loss, batch_loss)
                batch_scores_confidence = compute_quantiles(reference_confidence, batch_confidence)
                batch_scores_entropy = 1 - compute_quantiles(reference_entropy, batch_entropy)

                membership_scores[ptr:ptr+B] = (batch_scores_loss + batch_scores_confidence + batch_scores_entropy) / 3
                ptr += B

        logger.info("RMIA Ensemble Attack Completed")
        return membership_scores
