import time
import gc
import subprocess

from .experiment import Experiment
from .utils import set_seed
from .figures import (
    plot_metrics,
    plot_realizations,
    plot_realizations_spectra,
    plot_timeseries,
    plot_dailymax_timeseries,
    plot_rank_histogram,
    plot_pixelwise_statistics,
    plot_pixelwise_statistics_histogram,
    plot_time_avg_spectrum,
    plot_spectrogram,
)


start_time = time.time()

# plot the training and validation metrics
experiment = Experiment()
experiment.summary()

# Set random seed for reproducibility & select visible GPU
set_seed(experiment.seed)
if experiment.which_gpu is not None:
    subprocess.run(["export", f"CUDA_VISIBLE_DEVICES={experiment.which_gpu}"], shell=True)

# Produce plots
N = experiment.n_realizations
plot_metrics(experiment)

for var in experiment.predictand_names:

    plot_realizations(experiment, var=var, N=4, time_idx=-1)
    plot_realizations_spectra(experiment, var=var, N=N, time_idx=-1)
    plot_pixelwise_statistics(experiment, var=var, N=N, daily_max=False)
    plot_pixelwise_statistics(experiment, var=var, N=N, daily_max=True)
    plot_pixelwise_statistics_histogram(experiment, var=var, N=N, daily_max=False)
    plot_pixelwise_statistics_histogram(experiment, var=var, N=N, daily_max=True)
    plot_time_avg_spectrum(experiment, var=var, N=N, daily_max=False)
    plot_time_avg_spectrum(experiment, var=var, N=N, daily_max=True)
    plot_spectrogram(experiment, var=var, N=N, daily_max=False)
    plot_spectrogram(experiment, var=var, N=N, daily_max=True)

    for loc in experiment.loc_idxs:  # Location-specific plots

        plot_timeseries(experiment, var=var, N=N, xy=loc)
        plot_dailymax_timeseries(experiment, var=var, N=N, xy=loc)
        plot_rank_histogram(experiment, var=var, N=N, xy=loc, daily_max=False)
        plot_rank_histogram(experiment, var=var, N=N, xy=loc, daily_max=True)


gc.collect()
finish_time = time.time()
elapsed_time = finish_time - start_time
print(f"Finished in {elapsed_time/60:.2f} minutes.")
