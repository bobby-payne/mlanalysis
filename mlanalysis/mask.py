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


def apply_mask(data, mask):
    '''
    If data is a 2D tensor, then mask is applied to that tensor.
    If data is a 3D tensor, then it's assumed the first axis corresponds
    to different realizations of data, and the mask is applied to each
    realization along that axis.
    '''
    mask = mask.squeeze()
    mask = torch.where(mask, np.nan, 1)
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
