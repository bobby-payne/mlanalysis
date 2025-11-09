import os
import json
import torch
import numpy as np
from glob import glob

from .config import get_config
from .utils import transpose_nested_dict


def load_data ():

    # Read from config
    config = get_config()
    path_to_zarr = config['path_to_zarr']
    split = config['split']
    subset_idxs = config['subset_idxs']
    covariate_names = config['covariate_names']
    predictand_names = config['predictand_names']

    # Get PATHS to data (not the actual data yet)
    covariate_paths, groundtruth_paths = {}, {}
    if subset_idxs:
        i0, i1 = subset_idxs

    path_topography_lr = os.path.join(path_to_zarr, f"invariant/topography/lr_invariant/topography.pt")
    path_topography_hr = os.path.join(path_to_zarr, f"invariant/topography/hr_invariant/topography.pt")

    for name in covariate_names:
        covariate_paths[name] = sorted(glob(os.path.join(path_to_zarr, f"{split}/{name}/lr/*.pt")))[i0:i1]
    for name in predictand_names:
        groundtruth_paths[name] = sorted(glob(os.path.join(path_to_zarr, f"{split}/{name}/hr/*.pt")))[i0:i1]

    # Load data from paths
    covariate_data, groundtruth_data = {}, {}

    topography_lr = torch.load(path_topography_lr).float().unsqueeze(0).unsqueeze(0)
    topography_hr = torch.load(path_topography_hr).float().unsqueeze(0).unsqueeze(0)

    for name in covariate_names:
        covariate_data[name] = [torch.load(file).float().unsqueeze(0).unsqueeze(0) for file in covariate_paths[name]]
    for name in predictand_names:
        groundtruth_data[name] = [torch.load(file).float().unsqueeze(0).unsqueeze(0) for file in groundtruth_paths[name]]

    # Create a list of datetimes
    datetimes = []
    for file in covariate_paths[covariate_names[0]]:
        filename = os.path.basename(file)
        datetime_str = filename[-16:-3]
        datetimes.append(datetime_str)

    return covariate_data, groundtruth_data, topography_lr, topography_hr, datetimes


def load_min_max ():

    # Read from config
    config = get_config()
    path_to_zarr = config['path_to_zarr']
    split = config['split']
    covariate_names = config['covariate_names']
    predictand_names = config['predictand_names']

    covariate_data_min_max, groundtruth_data_min_max, topography_lr_min_max, topography_hr_min_max = {}, {}, {}, {}

    for name in covariate_names:
        with open(os.path.join(path_to_zarr, f"{name}_{split}_lr.zarr/zarr.json"), 'r') as metadata_file:
            metadata = json.load(metadata_file)
            data_min = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["min"]
            data_max = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["max"]
            covariate_data_min_max[name] = {"min": data_min, "max": data_max}
    for name in predictand_names:
        with open(os.path.join(path_to_zarr, f"{name}_{split}_hr.zarr/zarr.json"), 'r') as metadata_file:
            metadata = json.load(metadata_file)
            data_min = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["min"]
            data_max = metadata["consolidated_metadata"]["metadata"][name]["attributes"]["max"]
            groundtruth_data_min_max[name] = {"min": data_min, "max": data_max}

    with open(os.path.join(path_to_zarr, "topography_lr_invariant.zarr/zarr.json"), 'r') as metadata_file:
        metadata = json.load(metadata_file)
        topography_lr_min = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["min"]
        topography_lr_max = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["max"]
    topography_lr_min_max = {"min": topography_lr_min, "max": topography_lr_max}

    with open(os.path.join(path_to_zarr, "topography_hr_invariant.zarr/zarr.json"), 'r') as metadata_file:
        metadata = json.load(metadata_file)
        topography_hr_min = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["min"]
        topography_hr_max = metadata["consolidated_metadata"]["metadata"]["topography"]["attributes"]["max"]
    topography_hr_min_max = {"min": topography_hr_min, "max": topography_hr_max}

    covariate_data_min_max = transpose_nested_dict(covariate_data_min_max)  # TODO: fix
    groundtruth_data_min_max = transpose_nested_dict(groundtruth_data_min_max)

    return covariate_data_min_max, groundtruth_data_min_max, topography_lr_min_max, topography_hr_min_max


def load_model():

    # Read from config
    config = get_config()
    experiment_name = config['experiment_name']
    path_to_experiments = config['path_to_experiments']
    path_to_model = os.path.join(path_to_experiments, experiment_name, f"generator_{experiment_name}.pt")

    # Load model
    model = torch.jit.load(path_to_model).cuda()

    return model


def load_metrics():

    # Read from config
    config = get_config()
    experiment_name = config['experiment_name']
    path_to_experiments = config['path_to_experiments']
    path_to_metrics = os.path.join(path_to_experiments, experiment_name)

    # the time series of metrics will be stored in a nested dict
    metrics = ["mae", "wasserstein", "lsd"]
    splits = ["train", "validation"]
    metrics_dict = {s: {m: None for m in metrics} for s in splits}

    # Load metrics and store in nested dict
    for s in splits:
        for m in metrics:
            try:
                with open(os.path.join(path_to_metrics, f"{s}_{m}.json"), 'r') as f:
                    data = json.load(f)
                    metrics_dict[s][m] = (data[0]["x"], data[0]["y"])
            except Exception as e:
                print(f"Error loading {m.upper()} from split {s.upper()}: {e}")
                metrics_dict[s].pop(m, None)  # remove the key for that metric

    return metrics_dict
