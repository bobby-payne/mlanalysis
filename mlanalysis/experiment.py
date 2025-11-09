import torch
import numpy as np

from .config import get_config
from .load import load_data, load_min_max, load_model, load_metrics


class Experiment:

    def __init__(self):

        # Each config parameter becomes an attribute of the Experiment instance
        self.config = get_config()
        for key, value in self.config.data.items():
            setattr(self, key, value)

        # Data and their bounds become attributes of the Experiment instance
        data = load_data()
        data_min_max = load_min_max()
        self.data = {
            'covariates': data[0],
            'groundtruth': data[1],
            'topography_lr': data[2],
            'topography_hr': data[3],
        }
        self.timestamps = data[4]
        self.data_min = {
            'covariates': data_min_max[0]['min'],
            'groundtruth': data_min_max[1]['min'],
            'topography_lr': data_min_max[2]['min'],
            'topography_hr': data_min_max[3]['min'],
        }
        self.data_max = {
            'covariates': data_min_max[0]['max'],
            'groundtruth': data_min_max[1]['max'],
            'topography_lr': data_min_max[2]['max'],
            'topography_hr': data_min_max[3]['max'],
        }
        self.model = load_model()  # The actual PyTorch model
        self.metrics = load_metrics()  # Training and validation metrics

    def summary(self):

        print("Experiment Summary:")
        print("=====================")
        print("Configuration Parameters:")
        for key, value in self.config.data.items():
            print(f"  {key}: {value}")
        print("\nTraining Metrics:")
        for split, metrics in self.metrics.items():
            print(f"  {split}: {list(metrics.keys())}")
        print("\nData Shapes:")
        for key, value in self.data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"  {key} - {subkey}: {subvalue[0].shape}")
            else:
                print(f"  {key}: {value.shape}")
        print("\nData Bounds:")
        for key in self.data_min.keys():
            if isinstance(self.data_min[key], dict):
                for subkey in self.data_min[key].keys():
                    print(f"  {key} - {subkey} - min: {self.data_min[key][subkey]}, max: {self.data_max[key][subkey]}")
            else:
                print(f"  {key} - min: {self.data_min[key]}, max: {self.data_max[key]}")

    def realization(self, time_idx, seed=None):

        # set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate a realization using the model at the given time index
        model = self.model
        covariate_names = self.covariate_names
        covariate_data = self.data['covariates']
        topography_lr = self.data['topography_lr']
        topography_hr = self.data['topography_hr']
        
        input_tensors_lr = [covariate_data[name][time_idx] for name in covariate_names]
        input_tensor_lr = torch.cat(input_tensors_lr + [topography_lr], dim=1).cuda()  # Concatenate along channel dimension
        input_tensor_hr = topography_hr.cuda()
        with torch.no_grad():
            output_tensor = model(input_tensor_lr, input_tensor_hr)
            downscaled_field = output_tensor.squeeze().cpu()
        
        return downscaled_field
