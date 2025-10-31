import os
import json
import torch
import numpy as np
from glob import glob

from config import get_config


def load_data ():

    # Read from config
    config = get_config()
    path_to_model = config['path_to_model']
    base_path = config['base_path']
    split = config['split']
    subset_idxs = config['subset_idxs']
    covariate_names = config['covariates']
    predictand_names = config['predictands']

    # Get PATHS to data (not the actual data yet)
    covariate_paths, groundtruth_paths = {}, {}
    if subset_idxs:
        i0, i1 = subset_idxs
        N_samples = i1 - i0
    else:
        N_samples = len(glob(os.path.join(base_path, f"{split}/{covariate_names[0]}/lr/*.pt")))

    path_topography_lr = os.path.join(base_path, f"invariant/topography/lr_invariant/topography.pt")
    path_topography_hr = os.path.join(base_path, f"invariant/topography/hr_invariant/topography.pt")

    for name in covariate_names:
        covariate_paths[name] = sorted(glob(os.path.join(base_path, f"{split}/{name}/lr/*.pt")))[i0:i1]
    for name in predictand_names:
        groundtruth_paths[name] = sorted(glob(os.path.join(base_path, f"{split}/{name}/hr/*.pt")))[i0:i1]

    # Load data from paths (may take a while)
    covariate_data, groundtruth_data = {}, {}

    topography_lr = torch.load(path_topography_lr).float().unsqueeze(0).unsqueeze(0)
    topography_hr = torch.load(path_topography_hr).float().unsqueeze(0).unsqueeze(0)

    for name in covariate_names:
        covariate_data[name] = [torch.load(file).float().unsqueeze(0).unsqueeze(0) for file in covariate_paths[name]]
    for name in predictand_names:
        groundtruth_data[name] = [torch.load(file).float().unsqueeze(0).unsqueeze(0) for file in groundtruth_paths[name]]

    return covariate_data, groundtruth_data, topography_lr, topography_hr

def load_min_max ():

    # Read from config
    config = get_config()
    base_path = config['base_path']
    split = config['split']
    covariate_names = config['covariates']
    predictand_names = config['predictands']

    covariate_data_min_max, groundtruth_data_min_max = {}, {}

    # with open(os.path.join(base_path, "topography_lr_invariant.zarr/zarr.json"), 'r') as metadata_file:
    #     metadata = json.load(metadata_file)
    #     topography_lr_min = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["min"]
    #     topography_lr_max = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["max"]

    # with open(os.path.join(base_path, "topography_hr_invariant.zarr/zarr.json"), 'r') as metadata_file:
    #     metadata = json.load(metadata_file)
    #     topography_hr_min = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["min"]
    #     topography_hr_max = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["max"]

    for name in covariate_names:
        with open(os.path.join(base_path, f"{name}_{split}_lr.zarr/zarr.json"), 'r') as metadata_file:
            metadata = json.load(metadata_file)
            data_min = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["min"]
            data_max = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["max"]
            covariate_data_min_max[name] = {"min": data_min, "max": data_max}
    for name in predictand_names:
        with open(os.path.join(base_path, f"{name}_{split}_hr.zarr/zarr.json"), 'r') as metadata_file:
            metadata = json.load(metadata_file)
            data_min = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["min"]
            data_max = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["max"]
            groundtruth_data_min_max[name] = {"min": data_min, "max": data_max}

    return covariate_data_min_max, groundtruth_data_min_max


if __name__ == "__main__":
    print(load_data())
