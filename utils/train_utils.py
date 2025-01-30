import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.device_manager import get_device
from utils.data_loader import membership_collate_fn
from utils.logger import logger

def train_shadow_model(shadow_model: nn.Module, dataset: torch.utils.data.Dataset, lr: float = 0.01, epochs: int = 1) -> None:
    """
    Train a shadow model on the provided dataset.

    Args:
        shadow_model (nn.Module): The shadow model to train.
        dataset (torch.utils.data.Dataset): The dataset subset for training.
        lr (float, optional): Learning rate. Defaults to 0.01.
        epochs (int, optional): Number of training epochs. Defaults to 5.
    """
    device = get_device()
    shadow_model.to(device)
    optimizer = torch.optim.AdamW(shadow_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        collate_fn=membership_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    shadow_model.train()

    logger.info("Starting training of shadow model")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = shadow_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        avg_loss = epoch_loss / len(dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    logger.info("Shadow model training completed")
