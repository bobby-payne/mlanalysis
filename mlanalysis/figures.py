import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils import set_seed, compute_daily_maximum
from .spectral import get_rapsd, generate_realizations_spectra


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


def plot_realizations_spectra(experiment, var, time_idx, N, seed=None):

    # Generate realizations and get their spectra
    datetime_str = experiment.timestamps[time_idx]
    (realizations_spectra,
     realizations_mean_spectrum,
     bin_mids,
     _) = generate_realizations_spectra(
        experiment,
        time_idx,
        N,
        seed=seed
    )

    # ground truth spectrum
    groundtruth_data = experiment.data_scaled['groundtruth'][var][time_idx].squeeze()
    groundtruth_spectrum, _, _ = get_rapsd(groundtruth_data)

    # Figure constants
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8., 3.))
    ts = 6
    tick_fs = 8*0.6
    colors = ["royalblue"]*N + ["r", 'k']
    alphas = [0.4]*N + [1.0, 1.0]
    styles = ['-']*N + ['-.', '-']
    lws = [.6]*N + [1, 0.8]
    priority = [4]*N + [6, 5]
    field_names = [f"Realizations (N={N})"] + [None for i in range(N-1)] + ["Realizations Mean", "Ground Truth"]

    # plot
    ax[0].set_title("Radially-Averaged Power Spectral Density", fontsize=ts*1.5)
    ax[0].set_xlabel("Wavenumber (km$^{-1}$)", fontsize=ts)
    ax[0].set_ylabel("Power Density (km)", fontsize=ts)
    ax[0].grid(alpha=.3, which='major', ls='-')
    ax[0].grid(alpha=.1, which='minor', ls='--')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim(1/625, 1.5e-1)
    ax[0].tick_params(axis='both', labelsize=tick_fs)
    for i, spectrum in enumerate(realizations_spectra):
        ax[0].plot(bin_mids.numpy(), spectrum.numpy(), label=field_names[i], color=colors[i], linestyle=styles[i], lw=lws[i], alpha=alphas[i], zorder=priority[i])
    ax[0].plot(bin_mids.numpy(), realizations_mean_spectrum.numpy(), label=field_names[-2], color=colors[-2], linestyle=styles[-2], lw=lws[-2], alpha=alphas[-2], zorder=priority[-2])
    ax[0].plot(bin_mids.numpy(), groundtruth_spectrum.numpy(), label=field_names[-1], color=colors[-1], linestyle=styles[-1], lw=lws[-1], alpha=alphas[-1], zorder=priority[-1])
    ax[0].legend(fontsize=ts*1.1, frameon=False)
    ax[0].text(0.015, 0.08, f'{datetime_str[:10]} at {datetime_str[11:]}00', fontsize=1.5*ts, color='red', verticalalignment='top', horizontalalignment='left', transform=ax[0].transAxes)

    ax[1].set_title("Radially-Averaged Power Spectral Density (Normalized)", fontsize=ts*1.5)
    ax[1].set_xlabel("Wavenumber (km$^{-1}$)", fontsize=ts)
    ax[1].set_ylabel("Power Density (Normalized)", fontsize=ts)
    ax[1].grid(alpha=.3, which='major', ls='-')
    ax[1].grid(alpha=.1, which='minor', ls='--')
    ax[1].set_xscale('log')
    ax[1].set_xlim(1/625, 1.5e-1)
    ax[1].tick_params(axis='both', labelsize=tick_fs)
    for i, spectrum in enumerate(realizations_spectra):
        ax[1].plot(bin_mids.numpy(), spectrum.numpy()/groundtruth_spectrum.numpy(), label=field_names[i], color=colors[i], linestyle=styles[i], lw=lws[i], alpha=alphas[i], zorder=priority[i])
    ax[1].plot(bin_mids.numpy(), realizations_mean_spectrum.numpy()/groundtruth_spectrum.numpy(), label=field_names[-2], color=colors[-2], linestyle=styles[-2], lw=lws[-2], alpha=alphas[-2], zorder=priority[-2])
    ax[1].plot(bin_mids.numpy(), groundtruth_spectrum.numpy()/groundtruth_spectrum.numpy(), label=field_names[-1], color=colors[-1], linestyle=styles[-1], lw=lws[-1], alpha=alphas[-1], zorder=priority[-1])

    plt.tight_layout()
    _save_figure(fig, f"realizations_spectra_{var}_{datetime_str}", experiment)


def plot_timeseries(experiment, var, N, xy, seed=None):

    x, y = (xy[0], xy[1])  # the grid point to extract time series from
    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        seed=seed,
        unscale=True,
        round_negatives=False
    )[:, :, y, x]
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var][:, :, :, y, x].squeeze()

    # Plot each realization
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    for i in range(N):
        label = f"Realizations (N={N})" if i == 0 else None
        ax.plot(realizations_timeseries[:, i].numpy(), label=label, color='royalblue', lw=0.6, alpha=0.4)
    ax.plot(torch.mean(realizations_timeseries, dim=1).numpy(), label="Realizations Mean", color='red', lw=1.0, ls='--')
    ax.plot(groundtruth_timeseries.numpy(), label="Ground Truth", color='k', lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{var.upper()} Time Series at x = {x}, y = {y}", fontsize=12)
    ax.set_ylabel(f"{var.upper()}", fontsize=10)
    ax.grid(alpha=.3, which='major', ls='-')
    ax.set_xlim(0, len(experiment.timestamps)-1)
    xtick_idxs = [0, len(experiment.timestamps)//2, len(experiment.timestamps)-1]
    ax.set_xticks(xtick_idxs)
    ax.set_xticklabels([experiment.timestamps[i] for i in xtick_idxs])
    plt.tight_layout()
    _save_figure(fig, f"timeseries_{var}_x{x}y{y}", experiment)


def plot_dailymax_timeseries(experiment, var, N, xy, seed=None):

    if not experiment.timestamps[0][-2:] == '00':
        raise ValueError("First time index does not correspond to hour 00. Time series of daily maxima cannot be computed.")
    x, y = (xy[0], xy[1])  # the grid point to extract time series from
    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        seed=seed,
        unscale=True,
        round_negatives=False
    )[:, :, y, x]
    realizations_timeseries = compute_daily_maximum(realizations_timeseries, axis=0)
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var][:, :, :, y, x].squeeze()
    groundtruth_timeseries = compute_daily_maximum(groundtruth_timeseries, axis=0)
    N_days = realizations_timeseries.shape[0]

    # Plot each realization
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    for i in range(N):
        label = f"Realizations (N={N})" if i == 0 else None
        ax.plot(realizations_timeseries[:, i].numpy(), label=label, color='royalblue', lw=0.6, alpha=0.4)
    ax.plot(torch.mean(realizations_timeseries, dim=1).numpy(), label="Realizations Mean", color='red', lw=1.0, ls='--')
    ax.plot(groundtruth_timeseries.numpy(), label="Ground Truth", color='k', lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"{var.upper()} Time Series at x = {x}, y = {y}", fontsize=12)
    ax.set_ylabel(f"{var.upper()}", fontsize=10)
    ax.grid(alpha=.3, which='major', ls='-')
    ax.set_xlim(0, N_days-1)
    xtick_idxs = [0, N_days//2, N_days-1]
    print(xtick_idxs)
    ax.set_xticks(xtick_idxs)
    ax.set_xticklabels([experiment.timestamps[i*24][:10] for i in xtick_idxs])
    plt.tight_layout()
    _save_figure(fig, f"timeseries_dailymax_{var}_x{x}y{y}", experiment)
