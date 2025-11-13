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


def invert_feature_scaling(tensor, data_min, data_max, is_log_transformed=False):
    """Undo feature scaling to get back to original data range."""
    if is_log_transformed:
        epsilon = 1e-3
        return (1 + data_max/epsilon)**tensor
    else:
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


def compute_daily_maximum(tensor, axis=0):
    """
    Compute the daily maximum for a given tensor over a given axis.
    Assumes the first index of the tensor corresponds to hour 00;
    will give incorrect results if this is not the case.
    """

    n_days = tensor.shape[0] // 24
    daily_max = []
    for day_idx in range(n_days):
        start_idx = day_idx * 24
        end_idx = start_idx + 24
        daily_slice = torch.index_select(tensor, axis, torch.arange(start_idx, end_idx))
        daily_max.append(torch.max(daily_slice, dim=axis).values)
    daily_max = torch.stack(daily_max, dim=axis)
    return daily_max
