from torch.utils.data import Subset, Dataset
from typing import Optional, Any, List

from utils.logger import logger


class MembershipSubset(Subset):
    """
    A subset of a MembershipDataset, allowing for filtering specific indices.

    This class extends PyTorch's Subset class to provide additional functionality specific 
    to membership inference attacks.

    Attributes:
        dataset (Dataset): The complete dataset from which the subset is derived.
        transform (Optional[Any]): Optional transformations to apply to the subset data.
    """

    def __init__(self, dataset: Dataset, indices: List[int], transform: Optional[Any] = None):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.transform = transform
        logger.info(f"Created MembershipSubset with {len(self)} samples.")

    def __getitem__(self, idx: int) -> Any:
        data = self.dataset[self.indices[idx]]
        if self.transform:
            id_, img, label, membership = data
            img = self.transform(img)
            return id_, img, label, membership
        return data
