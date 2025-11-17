import time
import gc

from .experiment import Experiment
from .utils import set_seed
from .figures import (
    plot_metrics,
    plot_realizations,
    plot_realizations_spectra,
    plot_timeseries,
    plot_dailymax_timeseries,
    plot_pixelwise_statistics,
    plot_pixelwise_statistics_histogram,
    plot_time_avg_spectrum,
    plot_spectrogram,
)


start_time = time.time()

# plot the training and validation metrics
experiment = Experiment()
experiment.summary()

# Set random seed for reproducibility
set_seed(experiment.seed)

N = 1
loc = (84, 56)

# plot training and validation metrics
# plot_metrics(experiment)
# plot_realizations(experiment, var='fwi', N=4, time_idx=-1)
# plot_realizations_spectra(experiment, var='fwi', N=N, time_idx=-1)
# plot_timeseries(experiment, var='fwi', N=N, xy=loc)
# plot_dailymax_timeseries(experiment, var='fwi', N=N, xy=loc)
# plot_pixelwise_statistics(experiment, var='fwi', N=N, daily_max=False)
# plot_pixelwise_statistics(experiment, var='fwi', N=N, daily_max=True)
# plot_pixelwise_statistics_histogram(experiment, var='fwi', N=N, daily_max=False)
# plot_pixelwise_statistics_histogram(experiment, var='fwi', N=N, daily_max=True)
# plot_time_avg_spectrum(experiment, var='fwi', N=N, daily_max=False)
# plot_time_avg_spectrum(experiment, var='fwi', N=N, daily_max=True)
plot_spectrogram(experiment, var='fwi', N=N, daily_max=False)
plot_spectrogram(experiment, var='fwi', N=N, daily_max=True)


gc.collect()
finish_time = time.time()
elapsed_time = finish_time - start_time
print(f"Finished in {elapsed_time/60:.2f} minutes.")
