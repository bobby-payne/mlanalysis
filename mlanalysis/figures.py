import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils import set_seed


def _save_figure(fig, filename, experiment):

    path_to_output = experiment.path_to_output
    experiment_name = experiment.experiment_name
    format = experiment.output_fig_format
    dpi = experiment.output_fig_dpi
    path_to_output_full = os.path.join(
        path_to_output,
        experiment_name,
        f"{filename}.{format}"
    )
    os.makedirs(os.path.dirname(path_to_output_full), exist_ok=True)

    fig.savefig(
        path_to_output_full,
        bbox_inches='tight',
        dpi=dpi,
        format=format,
        facecolor='white',
    )
    plt.close(fig)


def plot_metrics(experiment):

    metrics_dict = experiment.metrics

    for i, m in enumerate(metrics_dict["train"].keys()):

        plt.figure(figsize=(8, 4), dpi=150)
        plt.plot(metrics_dict["train"][m][0], metrics_dict["train"][m][1], label=f"Training {m.upper()}", color='orange', lw=.6)
        plt.plot(metrics_dict["validation"][m][0], metrics_dict["validation"][m][1], label=f"Validation {m.upper()}", color='navy', lw=.6)
        plt.xlabel("Step")
        plt.yscale('log')
        if i == 0:
            plt.ylim(top=0.11)
        plt.legend()
        plt.grid(color='gray', ls='--', lw=.5, alpha=.15, which='both')
        _save_figure(plt.gcf(), f"training_{m}", experiment)

    plt.close('all')


def plot_realizations(experiment, var, N, time_idx, seed=None):

    # set seed for reproducibility and choose time index if necessary
    set_seed(seed)
    if time_idx is None:
        time_idx = np.random.randint(0, len(experiment.timestamps))

    # Generate realizations
    realizations = experiment.generate_realizations(time_idx=time_idx,
                                                    N_realizations=N,
                                                    seed=seed,
                                                    unscale=True,
                                                    round_negatives=False)
    groundtruth_data = experiment.data_scaled['groundtruth']
    datetime_str = experiment.timestamps[time_idx]

    # Figure constants
    fig, ax = plt.subplots(nrows=1, ncols=N+2, figsize=(8, 4), dpi=200)
    ts = 8
    vmin = 0
    vmax = 22
    cmap = 'viridis'

    # Plot the individual realizations
    for i in range(N):
        realization = realizations[i]
        ax[i].imshow(realization, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax[i].set_title(f"Realization {i+1}", fontsize=ts)
        ax[i].axis('off')

    # Plot the realization mean
    realization_mean = (1/N)*sum(realizations)
    ax[N].imshow(realization_mean, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax[N].set_title("Realization Mean", fontsize=ts)
    ax[N].axis('off')

    # Plot the ground truth
    ground_truth = groundtruth_data[var][time_idx].squeeze()
    refplot = ax[N+1].imshow(ground_truth, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax[N+1].set_title("Ground Truth", fontsize=ts)
    ax[N+1].axis('off')

    # Add datetime annotation... (11,3) for bottom right, (11,30) for top right
    ax[0].text(0, -5, f'{datetime_str[:10]} at {datetime_str[11:]}00', fontsize=.8*ts, color='red', verticalalignment='top', horizontalalignment='left')

    # Add colorbar (using the last plotted image for the colorbar)
    colorbar = fig.colorbar(refplot, ax=ax, orientation='horizontal', pad=0.05, aspect=140)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=5, width=.25)
    colorbar.ax.set_title(f"{var.upper()}",fontsize=ts, pad=2.5)

    _save_figure(fig, f"realizations_{var}_{datetime_str}", experiment)