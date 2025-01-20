import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from dataset import MembershipDataset, MembershipSubset, membership_collate_fn
from torchvision.models import resnet18
from tqdm import tqdm

# Automatically select device (GPU, MPS, or CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")

### Wrapper Membership Inference Attacks

def mia(func):
    def wrapper(model: Module, data: MembershipDataset):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument 'model' must be a PyTorch model (torch.nn.Module).")
        if not isinstance(data, MembershipDataset):
            raise TypeError("Argument 'data' must be an instance of MembershipDataset.")
        result = func(model, data)
        if not isinstance(result, np.ndarray):
            raise TypeError("The result must be a NumPy array.")
        if result.shape[0] != len(data):
            raise ValueError("The result must have the same size as the data.")
        return result
    return wrapper

### Random guess Membership Inference Attacks to test the pipeline

@mia
def random_guess(model, data):
    return np.random.rand(len(data))

### Likelihood Ratio Attack (LiRA)

@mia
def lira(model: Module, data: MembershipDataset, num_shadow_models=3, sample_size=10):
    """
    Implements the Likelihood Ratio Attack (LiRA) with per-example iteration.
    Args:
        model (torch.nn.Module): The target model.
        data (MembershipDataset): The membership dataset to evaluate.
        num_shadow_models (int): Number of shadow models to train.
        sample_size (int): Number of samples to draw for each shadow model.
    Returns:
        np.ndarray: The attack results, with values between 0 and 1.
    """
    # Placeholder for attack results
    attack_results = []

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Move models weights on device
    model.to(device)

    # Iterate over each example in the dataset
    for idx, (_, example_img, example_label, _) in tqdm(enumerate(data), total=len(data), desc="Processing Examples"):
        example_img, example_label = example_img.to(device), torch.tensor([example_label]).to(device)
        confsin, confsout = [], []

        # Train shadow models and collect statistics for the current example
        for _ in range(num_shadow_models):
            # Sample a shadow dataset
            indices = np.random.choice(len(data), size=sample_size, replace=False)

            # Train shadow in-model (including the example)
            shadow_in_dataset = MembershipSubset(data, indices.tolist() + [idx])  # Add the current example
            fin = train_shadow_model(model, shadow_in_dataset)
            confsin.append(compute_losses(fin, shadow_in_dataset, loss_fn))

            # Train shadow out-model (excluding the example)
            shadow_out_dataset = MembershipSubset(data, [i for i in indices if i != idx])  # Exclude the current example
            fout = train_shadow_model(model, shadow_out_dataset)
            confsout.append(compute_losses(fout, shadow_in_dataset, loss_fn))

        # Compute hyperparameters of the kernel
        mu_in, sigma_in = np.mean(confsin), np.std(confsin)
        mu_out, sigma_out = np.mean(confsout), np.std(confsout)

        # Compute the observed confidence score
        with torch.no_grad():
            model.eval()
            logits = model(example_img.unsqueeze(0))
            conf_obs = loss_fn(logits, example_label)

        conf_obs = conf_obs.cpu().numpy()

        # Compute the likelihood ratio
        p_in = gaussian_pdf(conf_obs, mu_in, sigma_in)
        p_out = gaussian_pdf(conf_obs, mu_out, sigma_out)
        likelihood_ratio = p_in / p_out

        # Store the result
        attack_results.append(likelihood_ratio)

    return np.array(attack_results)

def train_shadow_model(model: Module, dataset: MembershipSubset):
    """
    Train a shadow model on a given dataset.
    Args:
        model (torch.nn.Module): A PyTorch model.
        dataset (Subset): The dataset to train on.
    Returns:
        torch.nn.Module: The trained shadow model.
    """
    # Clone the model
    shadow_model = resnet18(weights=None)
    shadow_model.fc = torch.nn.Linear(512, 44)
    ckpt = torch.load("weights/01_MIA_67.pt", map_location="cpu")
    shadow_model.load_state_dict(ckpt)
    shadow_model.to(device)

    optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=membership_collate_fn, shuffle=True)

    # Training loop
    shadow_model.train()
    for epoch in range(5):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = shadow_model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

    return shadow_model

def compute_losses(model, dataset, loss_fn):
    """
    Compute losses for a dataset using a trained model.
    Args:
        model (torch.nn.Module): The trained model.
        dataset (Subset): The dataset to compute losses on.
        loss_fn (torch.nn.Module): The loss function.
    Returns:
        List[float]: List of loss values.
    """
    losses = []
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=membership_collate_fn, batch_size=32, shuffle=False)
    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            losses.extend(loss.cpu().numpy())
    return losses


def gaussian_pdf(x, mean, std):
    """
    Compute the Gaussian probability density function.
    Args:
        x (float): The observed value.
        mean (float): Mean of the distribution.
        std (float): Standard deviation of the distribution.
    Returns:
        float: The PDF value.
    """
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))


### Fast Likelihood Ratio Attack (LiRA)

@mia
def fast_lira(model: Module, data: MembershipDataset):
    # Placeholder for attack results
    attack_results = []

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Move models weights on device
    model.to(device)

    # Get the shadow dataset
    shadow_dataset: MembershipDataset = torch.load("data/pub.pt", weights_only=False)

    # Get indices for in-membership and out-membership samples
    shadow_in_indices = [i for i, member in enumerate(shadow_dataset.membership) if member == 1]
    shadow_out_indices = [i for i, member in enumerate(shadow_dataset.membership) if member == 0]
    
    # Create in and out datasets
    shadow_in_dataset = MembershipSubset(shadow_dataset, shadow_in_indices)
    shadow_out_dataset = MembershipSubset(shadow_dataset, shadow_out_indices)

    # Train the shadow in/out models
    fin = train_shadow_model(model, shadow_in_dataset)
    fout = train_shadow_model(model, shadow_out_dataset)

    # Compute the loss of the shadow models
    confsin = compute_losses(fin, shadow_in_dataset, loss_fn)
    confsout = compute_losses(fout, shadow_in_dataset, loss_fn)

    # Compute hyperparameters of the kernel
    mu_in, sigma_in = np.mean(confsin), np.std(confsin)
    mu_out, sigma_out = np.mean(confsout), np.std(confsout)

    # Iterate over each example in the dataset
    for idx, (_, example_img, example_label, _) in tqdm(enumerate(data), total=len(data), desc="Processing Examples"):
        example_img, example_label = example_img.to(device), torch.tensor([example_label]).to(device)

        # Compute the observed confidence score
        with torch.no_grad():
            model.eval()
            logits = model(example_img.unsqueeze(0))
            conf_obs = loss_fn(logits, example_label)

        conf_obs = conf_obs.cpu().numpy()

        # Compute the likelihood ratio
        p_in = gaussian_pdf(conf_obs, mu_in, sigma_in)
        p_out = gaussian_pdf(conf_obs, mu_out, sigma_out)
        likelihood_ratio = p_in / p_out

        # Store the result
        attack_results.append(likelihood_ratio)

    return np.array(attack_results).flatten()

### Robust Membership Inference Attack

@mia
def rmia(model, data):
    return np.random.rand(len(data))
