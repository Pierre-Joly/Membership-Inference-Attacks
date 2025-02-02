import os
import torch
import torch.nn as nn
from utils.data_loader import get_data_loader
from utils.device_manager import get_device
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.model_utils import clone_model

def train_shadow_model_common(model, subset, device, batch_size=256, num_workers=8, info_prefix=""):
    """
    Common training loop for a shadow model.
    
    Args:
        model (nn.Module): The model to train.
        subset (Dataset): The training subset.
        device (torch.device): The device to use.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers for DataLoader.
        info_prefix (str, optional): Prefix for logging output.
    
    Returns:
        dict: The trained model's state_dict.
    """
    model.train()
    dataloader = get_data_loader(dataset=subset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    epochs = 1
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        print(f"{info_prefix} Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(subset):.4f}")
        scheduler.step()
    
    return model.state_dict()


def ddp_worker(rank, world_size, subset, return_dict, model_index, batch_size=256):
    """
    DDP worker function to train one shadow model across multiple GPUs.
    
    Args:
        rank (int): Rank of the process.
        world_size (int): Total number of processes (e.g., 2).
        subset (Dataset): Training subset.
        return_dict (dict): A shared dictionary for returning the state_dict.
        model_index (int): Index of the shadow model.
        epochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size.
    """
    # Initialize the process group for DDP with NCCL backend (for CUDA)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set the device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create a new instance of the model and load the base state
    model = clone_model(device=device)
    
    # Wrap the model in DDP so that it uses the GPU corresponding to the rank
    ddp_model = DDP(model, device_ids=[rank])
    
    # Use the common training loop; include an info prefix for logging
    state_dict = train_shadow_model_common(
        model=ddp_model,
        subset=subset,
        device=device,
        batch_size=batch_size,
        num_workers=8,
        info_prefix=f"[Shadow Model {model_index} | DDP]"
    )
    
    # Only rank 0 returns the state_dict
    if rank == 0:
        return_dict[model_index] = state_dict
    
    dist.destroy_process_group()

def single_worker(subset, batch_size=256):
    """
    Single-device worker to train one shadow model.
    
    Args:
        subset (Dataset): Training subset.
        epochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size.
    
    Returns:
        dict: The trained model's state_dict.
    """
    device = get_device()
    model = clone_model(device=device)
    
    state_dict = train_shadow_model_common(
        model=model,
        subset=subset,
        device=device,
        batch_size=batch_size,
        num_workers=8,
        info_prefix="[Shadow Model Single]"
    )
    return state_dict

