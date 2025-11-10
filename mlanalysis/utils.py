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
