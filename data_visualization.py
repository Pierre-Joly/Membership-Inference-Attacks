import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import torch
from dataset import MembershipDataset
import torch.nn as nn

# Load the dataset
data: MembershipDataset = torch.load("data/priv_out.pt")
num_samples = 30
grid_cols = 6
grid_rows = (num_samples + grid_cols - 1) // grid_cols

# Random sampling indices
random_indices = np.random.choice(len(data), size=num_samples, replace=False)

# Load model
from torchvision.models import resnet18
model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 44)
ckpt = torch.load("weights/01_MIA_67.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Visualize random samples with loss
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(8, 8))

for idx, ax in zip(random_indices, axes.flat):
    id_, img, label, membership = data[idx]

    # Add batch dimension to the image and compute the model output
    img = img.unsqueeze(0)  # Shape becomes [1, C, H, W]
    with torch.no_grad():
        logits = model(img)
        label_tensor = torch.tensor([label])
        loss = loss_fn(logits, label_tensor).item()

    # Unnormalize and visualize the image
    img = img.squeeze(0)  # Remove batch dimension
    img = img.permute(1, 2, 0).numpy()  # Convert to HxWxC format
    
    ax.imshow(img)
    ax.set_title(f"ID: {id_}\nLabel: {label}\nMember: {membership}\nLoss: {loss:.2f}")
    ax.axis("off")

for ax in axes.flat[num_samples:]:
    ax.axis("off")
    
plt.tight_layout()
plt.show()
