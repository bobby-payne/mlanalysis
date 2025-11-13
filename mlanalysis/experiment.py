import torch
import numpy as np
from functools import cached_property, lru_cache

from .config import get_config
from .load import load_data, load_min_max, load_model, load_metrics
from .utils import invert_feature_scaling, set_seed


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

    @cached_property
    def data_scaled(self):
        '''
        Return data with feature scaling undone.
        TODO: precip is inverted incorrectly; fix this.
        '''
        data_scaled = {'covariates': {},
                       'groundtruth': {},
                       'topography_lr': {},
                       'topography_hr': {}}
        for name in self.data['covariates']:
            log_transformed = True if name == 'pr' else False
            data_scaled['covariates'][name] = invert_feature_scaling(
                self.data['covariates'][name],
                self.data_min['covariates'][name],
                self.data_max['covariates'][name],
                is_log_transformed=log_transformed
            )
        for name in self.data['groundtruth']:
            log_transformed = True if name == 'pr' else False
            data_scaled['groundtruth'][name] = invert_feature_scaling(
                self.data['groundtruth'][name],
                self.data_min['groundtruth'][name],
                self.data_max['groundtruth'][name],
                is_log_transformed=log_transformed
            )
        data_scaled['topography_lr'] = invert_feature_scaling(
            self.data['topography_lr'],
            self.data_min['topography_lr'],
            self.data_max['topography_lr'],
            is_log_transformed=False
        )
        data_scaled['topography_hr'] = invert_feature_scaling(
            self.data['topography_hr'],
            self.data_min['topography_hr'],
            self.data_max['topography_hr'],
            is_log_transformed=False
        )
        return data_scaled

    def summary(self):

        print("\nExperiment Summary:")
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
        print()

    def generate_realizations(self, time_idx, N_realizations, seed=None, unscale=True, round_negatives=False):

        # Set random seed for reproducibility
        set_seed(seed)

        # Prepare input tensors
        if time_idx is None:
            time_idx = np.random.randint(0, len(self.timestamps))
        model = self.model
        covariate_names = self.covariate_names
        covariate_data = self.data['covariates']
        topography_lr = self.data['topography_lr']
        topography_hr = self.data['topography_hr']
        input_tensors_lr = [covariate_data[name][time_idx] for name in covariate_names]
        input_tensor_lr = torch.cat(input_tensors_lr + [topography_lr], dim=1).cuda()  # Concatenate along channel dimension
        input_tensor_hr = topography_hr.cuda()

        # Generate a realization using the model at the given time index
        downscaled_fields = []
        with torch.no_grad():
            for _ in range(N_realizations):
                output_tensor = model(input_tensor_lr, input_tensor_hr)
                downscaled_field = output_tensor.squeeze().cpu()
                downscaled_fields.append(downscaled_field)
        downscaled_fields = torch.stack(downscaled_fields, dim=0)
        if round_negatives:
            downscaled_fields = torch.clamp(downscaled_fields, min=0.0)

        # remove feature scaling
        # TODO: will need to modify if multiple predictands
        # current behaviour applies same inverse scaling to all predictands
        if unscale:
            for predictand in self.predictand_names:
                log_transformed = True if predictand == 'pr' else False
                downscaled_fields = invert_feature_scaling(
                    downscaled_fields,
                    self.data_min['groundtruth'][predictand],
                    self.data_max['groundtruth'][predictand],
                    is_log_transformed=log_transformed,
                )

        return downscaled_fields

    @lru_cache(maxsize=None)
    def generate_realization_timeseries(self, N_realizations, seed=None, unscale=True, round_negatives=False):
        '''
        Generate realizations for all time indices.
        Results are cached for each unique set of arguments using functools.lru_cache.
        Returns as a tensor with the first axis as time,
        second axis as realization number.
        '''
        realizations_timeseries = []
        for time_idx in range(len(self.timestamps)):
            realizations = self.generate_realizations(
                time_idx=time_idx,
                N_realizations=N_realizations,
                seed=seed,
                unscale=unscale,
                round_negatives=round_negatives,
            )
            realizations_timeseries.append(realizations)
        return torch.stack(realizations_timeseries, axis=0)
