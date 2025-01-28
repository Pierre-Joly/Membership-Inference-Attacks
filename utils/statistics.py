import numpy as np
import torch
from math import sqrt

def gaussian_pdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Compute the Gaussian probability density function for an array of values.

    Args:
        x (np.ndarray): Observed values.
        mean (np.ndarray): Mean of the Gaussian distribution.
        std (np.ndarray): Standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: The PDF values for each element in x.
    """
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def gaussian_cdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gaussian cumulative distribution function for a tensor of values.

    Args:
        x (torch.Tensor): Observed values.
        mu (torch.Tensor): Mean of the Gaussian distribution.
        sigma (torch.Tensor): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The CDF values for each element in x.
    """
    z = (x - mu) / (sigma + 1e-9)
    return 0.5 * (1 + torch.erf(z / sqrt(2)))

def compute_quantiles(sorted_arr, measures):
    """
    Compute the quantile values for a set of measures based on a sorted reference array.

    Args:
        sorted_arr (numpy.ndarray or list): A sorted array that serves as the reference
            distribution. This array must be sorted in ascending order.
        measures (numpy.ndarray or list): An array of values for which to compute 
            quantile scores based on the reference array.

    Returns:
        numpy.ndarray: An array of quantile scores, where each value is between 0 and 1, 
        representing the proportion of elements in `sorted_arr` that are less than or 
        equal to the corresponding element in `measures`.
    """
    indices = np.searchsorted(sorted_arr, measures, side='right')
    return indices / len(sorted_arr)
