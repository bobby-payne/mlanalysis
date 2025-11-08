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
