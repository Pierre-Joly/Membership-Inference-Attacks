import torch
from torch.utils.data import ConcatDataset

from datasets.dataset import MembershipDataset
from datasets.subset import MembershipSubset
from utils.logger import logger

def get_out_dataset(reference_data: str) -> MembershipSubset:
    """
    Get out-of-distribution data.

    Args:
        reference_data (str): Path to the public data.

    Returns:
        MembershipSubset: Subset of non-member data.
    """
    logger.info(f"Loading public data from {reference_data}")
    data_pub: MembershipDataset = torch.load(reference_data, map_location='cpu', weights_only=False)
    non_member_indices = [i for i, m in enumerate(data_pub.membership) if m == 0]
    data_out = MembershipSubset(data_pub, non_member_indices)
    logger.info(f"Extracted {len(data_out)} non-member samples")
    return data_out

def get_shadow_dataset(data_priv: MembershipDataset, reference_data: str) -> ConcatDataset:
    """
    Combine private and public data to create a shadow dataset.

    Args:
        data_priv (MembershipDataset): Private data.
        data_pub (MembershipDataset): Public data.

    Returns:
        ConcatDataset: Combined dataset.
    """
    data_out = get_out_dataset(reference_data)
    return ConcatDataset([data_priv, data_out])
