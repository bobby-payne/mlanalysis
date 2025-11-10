import time

from .experiment import Experiment
from .figures import plot_metrics, plot_realizations


start_time = time.time()

# plot the training and validation metrics
experiment = Experiment()
experiment.summary()

# plot training and validation metrics
plot_metrics(experiment)
plot_realizations(experiment, var='fwi', N=4, time_idx=None, seed=242)

finish_time = time.time()
elapsed_time = finish_time - start_time
print(f"Finished in {elapsed_time/60:.2f} minutes.")
