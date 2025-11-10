import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def transpose_nested_dict(nested_dict):
    """
    Transpose a nested dictionary.

    Args:
        nested_dict (dict): A nested dictionary of the form {key1: {key2: value}}.

    Returns:
        dict: A transposed nested dictionary of the form {key2: {key1: value}}.
    """

    nested_dict_T = {
        key2: {key1: nested_dict[key1][key2] for key1 in nested_dict}
        for key2 in next(iter(nested_dict.values()))
    }
    return nested_dict_T


def invert_feature_scaling(tensor, data_min, data_max):
    """Undo feature scaling to get back to original data range."""
    return tensor * (data_max - data_min) + data_min


def compute_statistics(data, prestacked=True, axis=0):
    """
    Compute statistics for a data tensor over the given axis.
    Returns the stats as a tuple in the order:
    (mean, standard deviation, median, 1st perc., 95th perc., 99th perc.)
    """

    if not prestacked:
        data = np.stack(data, axis=axis)

    data_mean = np.nanmean(data, axis=axis)
    data_std = np.nanstd(data, axis=axis)
    data_median = np.nanmedian(data, axis=axis)
    data_1p = np.nanpercentile(data, q=1, axis=axis)
    data_95p = np.nanpercentile(data, q=95, axis=axis)
    data_99p = np.nanpercentile(data, q=99, axis=axis)

    return (data_mean, data_std, data_median, data_1p, data_95p, data_99p)
