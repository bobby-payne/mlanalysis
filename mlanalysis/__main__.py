import numpy as np

from .experiment import Experiment
from .figures import plot_metrics


# plot the training and validation metrics
experiment = Experiment()
experiment.summary()

# plot training and validation metrics
plot_metrics(experiment)
