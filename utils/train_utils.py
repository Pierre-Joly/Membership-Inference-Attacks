import torch
import torch.nn as nn
from utils.data_loader import get_data_loader
from utils.device_manager import get_device
from utils.model_utils import clone_model

def train_shadow_model_common(model, subset_train, subset_val, device, batch_size=256, num_workers=8, info_prefix=""):
    """
    Common training loop for a shadow model with a validation-based learning rate scheduler.
    
    Args:
        model (nn.Module): The model to train.
        subset_train (Dataset): The training subset.
        subset_val (Dataset): The validation subset.
        device (torch.device): The device to use.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers for DataLoader.
        info_prefix (str, optional): Prefix for logging output.
    
    Returns:
        dict: The trained model's state_dict.
    """
    model.to(device)
    model.train()
    
    # Training and validation dataloaders
    train_loader = get_data_loader(dataset=subset_train, batch_size=batch_size, shuffle=True)
    val_loader = get_data_loader(dataset=subset_val, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    lr_decay_factor = 0.1  # Factor to reduce the learning rate
    patience_counter = 0
    
    epochs = 60  # Number of epochs
    for epoch in range(epochs):
        # Training loop
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        
        # Normalize losses
        epoch_loss /= len(subset_train)
        val_loss /= len(subset_val)
        
        # Print epoch results
        print(f"{info_prefix} Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Adjust learning rate if validation loss plateaus
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                new_lr = optimizer.param_groups[0]['lr'] * lr_decay_factor
                if new_lr < 1e-6:  # Stop reducing if the learning rate becomes too small
                    print(f"{info_prefix} Learning rate reached minimum threshold.")
                    break
                print(f"{info_prefix} Reducing learning rate to {new_lr:.6f}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                patience_counter = 0
    
    return model.state_dict()


def single_worker(subset_train, subset_val, batch_size=256):
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
        subset_train=subset_train,
        subset_val=subset_val,
        device=device,
        batch_size=batch_size,
        num_workers=8,
        info_prefix="[Shadow Model Single]"
    )
    return state_dict

