import torch

def get_device() -> torch.device:
    """
    Automatically select device (GPU, MPS, or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
