import torch
import pandas as pd
from dataset import MembershipDataset
from mia import random_guess, lira, fast_lira

#### LOADING THE MODEL

from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("weights/01_MIA_67.pt", map_location="cpu")
model.load_state_dict(ckpt)

#### DATASETS

data: MembershipDataset = torch.load("data/priv_out.pt", weights_only=False)
    
#### EXAMPLE SUBMISSION

score = lira(model, data)

df = pd.DataFrame(
    {
        "ids": data.ids,
        "score": score,
    }
)

df.to_csv("submission.csv", index=None)
