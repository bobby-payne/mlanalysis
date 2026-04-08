import time
import numpy as np
import torch


def get_mask(tensor, val=0.0):
    '''
    Compute a mask for a given tensor. By default, mask has value False where
    tensor=val, and True otherwise.
    '''

    if np.isnan(val):
        mask = ~torch.isnan(tensor)
    else:
        mask = torch.where(tensor == val, True, False)

    return mask


def apply_mask_spatial(data, mask):
    '''
    If data is a 2D tensor, then mask is applied to that tensor.
    If data is a 3D tensor, then it's assumed the first axis corresponds
    to different realizations of data, and the mask is applied to each
    realization along that axis.
    '''
    mask = mask.squeeze()
    mask = torch.where(mask, torch.nan, 1)
    if data.shape.__len__() == 2:
        data = data * mask
    elif data.shape.__len__() == 3:
        n_realizations = data.shape[0]
        mask = mask.repeat(n_realizations, 1, 1)
        data = data * mask
    else:
        raise IndexError(
            f"Input data shape must be a 2D or 3D tensor. Received shape {data.shape}."
        )

    return data


def apply_mask_timeseries(data, mask):
    '''
    Apply a 1D mask to a 1D timeseries (data).
    If data is 2D, then the mask is applied by iterating
    over the first axis. (i.e., the first axis must be the time axis.)
    '''

    mask = mask.squeeze()
    assert mask.shape.__len__() == 1, "Mask is not 1D."
    mask = torch.where(mask, torch.nan, 1)

    if data.shape.__len__() == 1:
        data = data * mask
    elif data.shape.__len__() == 2:
        n_realizations = data.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_realizations)
        data = data * mask
    else:
        raise IndexError(
            f"Input data shape must be a 1D or 2D tensor. Received shape {data.shape}."
        )

    return data


def apply_mask_spatial_timeseries(data, mask):
    """Apply a time-varying mask to a time series of 2D fields.
    If data is 3D, then the assumed dims are (N_time * N_y * N_x)
    If data is 4D, then the assumed dims are (N_time * N_real * N_y * N_x)"""

    mask = mask.squeeze()
    assert mask.shape.__len__() == 3, "Mask is not 3D."
    mask = torch.where(mask, torch.nan, 1)

    if data.shape.__len__() == 3:
        data = data * mask
    elif data.shape.__len__() == 4:
        n_realizations = data.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_realizations, 1, 1)
        data = data * mask
    else:
        raise IndexError(
            f"Input data shape must be a 3D or 4D tensor. Received shape {data.shape}."
        )

    return data
