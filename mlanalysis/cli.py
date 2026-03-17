import os
import time
import gc

from .config import get_config


def main():

    start_time = time.time()

    try:

        # Select visible GPUs (must occur before calling Experiment class)
        which_gpu = get_config()['which_gpu']
        if which_gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)

        # Import modules with Torch dependency
        # (must be AFTER setting CUDA_VISIBLE_DEVICES)
        from .experiment import Experiment
        from .utils import set_seed
        from .figures import (
            plot_metrics,
            plot_realizations,
            plot_realizations_spectra,
            plot_timeseries,
            plot_dailymax_timeseries,
            plot_pixelwise_statistics,
            plot_time_avg_spectrum,
            plot_spectrogram,
        )

        # Set random seed for reproducibility
        set_seed(get_config()['seed'])

        # Loop over years (if multiple) and generate plots for each year
        for year in get_config()['years']:

            # Plot the training and validation metrics
            experiment = Experiment(year=year)
            experiment.summary()

            # Generate plots
            print("Generating plots...")
            N = experiment.n_realizations
            plot_metrics(experiment)

            for var in experiment.predictand_names:

                plot_realizations(experiment, var=var, N=4, time_idx=-1)
                plot_realizations_spectra(experiment, var=var, N=N, time_idx=-1)
                plot_pixelwise_statistics(experiment, var=var, N=N, daily_max=False)
                plot_pixelwise_statistics(experiment, var=var, N=N, daily_max=True)
                plot_time_avg_spectrum(experiment, var=var, N=N, daily_max=False)
                plot_time_avg_spectrum(experiment, var=var, N=N, daily_max=True)
                plot_spectrogram(experiment, var=var, N=N, daily_max=False)
                plot_spectrogram(experiment, var=var, N=N, daily_max=True)

                for loc in experiment.loc_idxs:  # Location-specific plots

                    plot_timeseries(experiment, var=var, N=N, xy=loc)
                    plot_dailymax_timeseries(experiment, var=var, N=N, xy=loc)

    finally:

        gc.collect()
        elapsed_time = time.time() - start_time
        print(f"Finished in {elapsed_time/60:.2f} minutes.")
