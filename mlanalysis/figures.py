import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .utils import compute_daily_maximum, compute_statistics
from .spectral import get_rapsd, compute_realizations_spectra


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


def plot_realizations(experiment, var, N, time_idx, vmin=0, vmax=30):

    # choose time index if necessary
    if time_idx is None:
        time_idx = np.random.randint(0, len(experiment.timestamps))

    # Generate realizations
    realizations = experiment.generate_realizations(time_idx=time_idx,
                                                    N_realizations=N,
                                                    unscale=True,
                                                    round_negatives=False)
    groundtruth_data = experiment.data_scaled['groundtruth']
    datetime_str = experiment.timestamps[time_idx]

    # Figure constants
    fig, ax = plt.subplots(nrows=1, ncols=N+2, figsize=(8, 4))
    ts = 8
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

    _save_figure(fig, f"{var}_realizations_{datetime_str}", experiment)


def plot_realizations_spectra(experiment, var, time_idx, N):

    # Generate realizations and get their spectra
    datetime_str = experiment.timestamps[time_idx]
    realizations = experiment.generate_realizations(
            time_idx=time_idx,
            N_realizations=N,
            unscale=True,
            round_negatives=False,
        )
    (realizations_spectra,
     realizations_mean_spectrum,
     bin_mids,
     _) = compute_realizations_spectra(
        experiment,
        realizations,
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
    _save_figure(fig, f"{var}_realizations_spectra_{datetime_str}", experiment)


def plot_timeseries(experiment, var, N, xy):

    x, y = (xy[0], xy[1])  # the grid point to extract time series from
    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        unscale=True,
        round_negatives=False
    )[:, :, y, x]
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var][:, :, :, y, x].squeeze()

    # Plot each realization
    fig, ax = plt.subplots(figsize=(10, 4))
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
    _save_figure(fig, f"{var}_timeseries_x{x}y{y}", experiment)


def plot_dailymax_timeseries(experiment, var, N, xy):

    if not experiment.timestamps[0][-2:] == '00':
        raise ValueError("First time index does not correspond to hour 00. Time series of daily maxima cannot be computed.")
    x, y = (xy[0], xy[1])  # the grid point to extract time series from
    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        unscale=True,
        round_negatives=False
    )[:, :, y, x]
    realizations_timeseries = compute_daily_maximum(realizations_timeseries, axis=0)
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var][:, :, :, y, x].squeeze()
    groundtruth_timeseries = compute_daily_maximum(groundtruth_timeseries, axis=0)
    N_days = realizations_timeseries.shape[0]

    # Plot each realization
    fig, ax = plt.subplots(figsize=(10, 4))
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
    _save_figure(fig, f"{var}_timeseries_dailymax_x{x}y{y}", experiment)


def plot_pixelwise_statistics(experiment, var, N, daily_max=False):

    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        unscale=True,
        round_negatives=True,
    )
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var].squeeze()
    if daily_max:
        realizations_timeseries = compute_daily_maximum(realizations_timeseries, axis=0)
        groundtruth_timeseries = compute_daily_maximum(groundtruth_timeseries, axis=0)

    # Compute statistics
    (mean_field_sr,
     std_field_sr,
     median_field_sr,
     _,
     p95_field_sr,
     p99_field_sr) = compute_statistics(realizations_timeseries, prestacked=True, axis=0)

    (mean_field_gt,
     std_field_gt,
     median_field_gt,
     _,
     p95_field_gt,
     p99_field_gt) = compute_statistics(groundtruth_timeseries, prestacked=True, axis=0)

    # Average over realizations and plot statistics
    stats = {
        'Mean': (mean_field_sr, mean_field_gt),
        'Standard Deviation': (std_field_sr, std_field_gt),
        'Median': (median_field_sr, median_field_gt),
        '95 Percentile': (p95_field_sr, p95_field_gt),
        '99 Percentile': (p99_field_sr, p99_field_gt)
    }
    stats = {statname: (np.nanmean(fields[0], axis=0), fields[1]) for statname, fields in stats.items()}

    cmap = 'viridis'
    ts = 5
    for stat_name, stat_fields in stats.items():

        vmin = min(np.nanmin(stat_fields[0]), np.nanmin(stat_fields[1])).item()
        vmax = max(np.nanmax(stat_fields[0]), np.nanmax(stat_fields[1])).item()
        dvmin = np.nanmin(stat_fields[0] - stat_fields[1]).item()
        dvmax = np.nanmax(stat_fields[0] - stat_fields[1]).item()

        if np.abs(dvmin) < np.abs(dvmax):
            dvmin = -dvmax
        elif np.abs(dvmin) > np.abs(dvmax):
            dvmax = -dvmin

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6, 2.5))
        for axis in ax:
            axis.axis('off')

        if daily_max:
            fig.suptitle(f"{stat_name.title()} Daily Maximum {var.upper()}", fontsize=ts*1.2, y=0.95)
        else:
            fig.suptitle(f"{stat_name.title()} {var.upper()}", fontsize=ts*1.2, y=0.95)
        ax[0].set_title(f"Downscaled", size=ts)
        ax[0].imshow(stat_fields[0], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        ax[1].set_title(f"Ground Truth", size=ts)
        plot1 = ax[1].imshow(stat_fields[1], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        ax[2].set_title(f"Downscaled - Ground Truth",size=ts)
        plot2 = ax[2].imshow(stat_fields[0] - stat_fields[1], cmap='RdBu_r', vmin=dvmin, vmax=dvmax, origin='lower')

        # Add colorbar (using the last plotted image for the colorbar)
        colorbar = fig.colorbar(plot1, ax=ax[:2], orientation='horizontal', pad=0.05, aspect=80)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=ts, width=.25)
        colorbar.ax.set_title(None)

        colorbar_dif = fig.colorbar(plot2, ax=ax[2], orientation='horizontal', pad=0.05, aspect=25)
        colorbar_dif.outline.set_visible(False)
        colorbar_dif.ax.tick_params(labelsize=ts, width=.25) 
        colorbar_dif.ax.set_title(None)

        if daily_max:
            _save_figure(fig, f"{var}_dailymax_pixelwise_{stat_name.lower().replace(' ', '_')}", experiment)
        else:
            _save_figure(fig, f"{var}_pixelwise_{stat_name.lower().replace(' ', '_')}", experiment)


def plot_pixelwise_statistics_histogram(experiment, var, N, daily_max=False):

    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        unscale=True,
        round_negatives=True,
    )
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var].squeeze()
    if daily_max:
        realizations_timeseries = compute_daily_maximum(realizations_timeseries, axis=0)
        groundtruth_timeseries = compute_daily_maximum(groundtruth_timeseries, axis=0)

    # Compute statistics
    (mean_field_sr,
     std_field_sr,
     median_field_sr,
     p5_field_sr,
     p95_field_sr,
     p99_field_sr) = compute_statistics(realizations_timeseries, prestacked=True, axis=0)

    (mean_field_gt,
     std_field_gt,
     median_field_gt,
     p5_field_gt,
     p95_field_gt,
     p99_field_gt) = compute_statistics(groundtruth_timeseries, prestacked=True, axis=0)

    stats = {
        'Mean': (mean_field_sr, mean_field_gt),
        'Standard Deviation': (std_field_sr, std_field_gt),
        'Median': (median_field_sr, median_field_gt),
        '5 Percentile': (p5_field_sr, p5_field_gt),
        '95 Percentile': (p95_field_sr, p95_field_gt),
        '99 Percentile': (p99_field_sr, p99_field_gt)
    }

    # Plot statistics
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4), dpi=200)
    ts = 5
    nbins = 200

    for i, axis in enumerate(ax.flat):

        statname = list(stats.keys())[i]
        min_val = np.nanmin([stats[statname][0].min(), stats[statname][1].min()])
        max_val = np.nanmax([stats[statname][0].max(), stats[statname][1].max()])
        bins = np.linspace(min_val, max_val, nbins)
        axis.hist(stats[statname][1].flatten(), bins=bins, alpha=0.5, label='HR Truth (WRF)', density=0)
        axis.hist(stats[statname][0].flatten(), bins=bins, alpha=0.5, label='HR Downscaled', color='orange', density=0)
        axis.set_title(statname, size=ts)

    subtitle = f"Daily Maximum {var.upper()}" if daily_max else var.upper()
    fig.suptitle(f"Spatial Distributions of Pixelwise Summary Statistics\n({subtitle})", fontsize=8, fontweight='bold', y=0.98)

    for i, axes in enumerate(ax.flat):
        if i == 0:
            axes.legend(fontsize=5, frameon=False, loc='upper right')
        else:
            axes.legend(fontsize=5, frameon=False)
        axes.tick_params(axis='both', which='major', labelsize=3.5)
        axes.title.set_fontweight('bold')
        if not i == 0:
            axes.set_xlabel(var.upper(), size=.7*ts)

    plt.subplots_adjust(hspace=0.34, wspace=0.14)

    if daily_max:
        _save_figure(fig, f"{var}_dailymax_pixelwise_histogram", experiment)
    else:
        _save_figure(fig, f"{var}_pixelwise_histogram", experiment)


def plot_time_avg_spectrum(experiment, var, N, daily_max=False):

    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        unscale=True,
        round_negatives=False,
    )
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var].squeeze()
    if daily_max:
        realizations_timeseries = compute_daily_maximum(realizations_timeseries, axis=0)
        groundtruth_timeseries = compute_daily_maximum(groundtruth_timeseries, axis=0)
    N_time = realizations_timeseries.shape[0]

    # Compute time-averaged spectra
    spectra_realizations_all, spectra_groundtruth_all = [], []
    for i in range(N_time):

        # realization spectra, averaged over the N realizations
        realizations_spectra = compute_realizations_spectra(
            realizations_timeseries[i],
            d=4.0,
            )[0]
        spectra_realizations_all.append(torch.mean(torch.stack(realizations_spectra, axis=0), axis=0))

        # ground truth spectrum
        groundtruth_spectrum, _, _ = get_rapsd(groundtruth_timeseries[i], d=4.0)
        spectra_groundtruth_all.append(groundtruth_spectrum)
        if i == N_time - 1:
            bin_mids = get_rapsd(groundtruth_timeseries[i], d=4.0)[1]

    # average over all time steps
    spectra_realizations_avg = torch.mean(torch.stack(spectra_realizations_all, axis=0), axis=0)
    spectra_groundtruth_avg = torch.mean(torch.stack(spectra_groundtruth_all, axis=0), axis=0)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8.,3.))
    ts = 6
    tick_fs = 8*0.6
    colors = ["r", 'k']
    alphas = [1.0, 1.0]
    styles = ['-.', '-']
    lws = [1, 0.8]
    priority = [6, 5]

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
    ax[0].plot(bin_mids[1:].numpy(), spectra_realizations_avg[1:].numpy(), label="Super-Resolved", color=colors[0], linestyle=styles[0], lw=lws[0], alpha=alphas[0], zorder=priority[0])
    ax[0].plot(bin_mids[1:].numpy(), spectra_groundtruth_avg[1:].numpy(), label="Ground Truth", color=colors[1], linestyle=styles[1], lw=lws[1], alpha=alphas[1], zorder=priority[1])
    ax[0].legend(fontsize=ts*1.1, frameon=False)

    ax[1].set_title("Radially-Averaged Power Spectral Density (Normalized)", fontsize=ts*1.5)
    ax[1].set_xlabel("Wavenumber (km$^{-1}$)", fontsize=ts)
    ax[1].set_ylabel("Power Density (Normalized)", fontsize=ts)
    ax[1].grid(alpha=.3, which='major', ls='-')
    ax[1].grid(alpha=.1, which='minor', ls='--')
    ax[1].set_xscale('log')
    ax[1].set_xlim(1/625, 1.5e-1)
    ax[1].tick_params(axis='both', labelsize=tick_fs)
    ax[1].plot(bin_mids[1:].numpy(), spectra_realizations_avg[1:].numpy()/spectra_groundtruth_avg[1:].numpy(), label="Super-Resolved", color=colors[0], linestyle=styles[0], lw=lws[0], alpha=alphas[0], zorder=priority[0])
    ax[1].plot(bin_mids[1:].numpy(), spectra_groundtruth_avg[1:].numpy()/spectra_groundtruth_avg[1:].numpy(), label="Ground Truth", color=colors[1], linestyle=styles[1], lw=lws[1], alpha=alphas[1], zorder=priority[1])
    # ax[1].legend(fontsize=ts*1.1, frameon=False)

    plt.tight_layout()
    if daily_max:
        _save_figure(fig, f"{var}_dailymax_time_avg_spectrum", experiment)
    else:
        _save_figure(fig, f"{var}_time_avg_spectrum", experiment)


def plot_spectrogram(experiment, var, N, daily_max=False):

    realizations_timeseries = experiment.generate_realization_timeseries(
        N_realizations=N,
        unscale=True,
        round_negatives=False,
    )
    groundtruth_timeseries = experiment.data_scaled['groundtruth'][var].squeeze()
    if daily_max:
        realizations_timeseries = compute_daily_maximum(realizations_timeseries, axis=0)
        groundtruth_timeseries = compute_daily_maximum(groundtruth_timeseries, axis=0)
    N_time = realizations_timeseries.shape[0]

    # Compute spectra
    spectra_realizations_all, spectra_groundtruth_all = [], []
    for i in range(N_time):

        # realization spectra, averaged over the N realizations
        realizations_spectra = compute_realizations_spectra(
            realizations_timeseries[i],
            d=4.0,
            )[0]
        spectra_realizations_all.append(torch.mean(torch.stack(realizations_spectra, axis=0), axis=0))

        # ground truth spectrum
        groundtruth_spectrum, _, _ = get_rapsd(groundtruth_timeseries[i], d=4.0)
        spectra_groundtruth_all.append(groundtruth_spectrum)

    # Plot
    datetime_monthstart_idxs = []
    for i, datetime in enumerate(experiment.timestamps):
        if datetime[-5:] == '01-00':
            if daily_max:
                datetime_monthstart_idxs.append(i // 24)
            else:
                datetime_monthstart_idxs.append(i)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14, 8), dpi=150, sharex=True)
    plt.subplots_adjust(hspace=0.)
    cmap = 'magma'
    spectra_realizations_all_numpy = torch.stack(spectra_realizations_all, axis=0).numpy().T
    spectra_groundtruth_all_numpy = torch.stack(spectra_groundtruth_all, axis=0).numpy().T
    min_val = min(spectra_realizations_all_numpy[1:].min(), spectra_groundtruth_all_numpy[1:].min())
    max_val = max(spectra_realizations_all_numpy[1:].max(), spectra_groundtruth_all_numpy[1:].max())
    print(spectra_groundtruth_all_numpy[1:].shape, spectra_realizations_all_numpy[1:].shape)

    ax[0].set_title("Spectrogram of Radially-Averaged Power Spectral Density", fontsize=8)
    plot0 = ax[0].imshow(spectra_realizations_all_numpy[1:], aspect='auto', origin='lower', cmap=cmap, norm=LogNorm(vmin=min_val, vmax=max_val))
    ax[0].set_ylabel("Wavenumber (cycles)", fontsize=7)
    ax[0].set_xticks(datetime_monthstart_idxs)
    ax[0].text(0.99, 0.95, "Super-Resolved", fontsize=14, color='w', verticalalignment='top', horizontalalignment='right', transform=ax[0].transAxes)

    ax[1].imshow(spectra_groundtruth_all_numpy[1:], aspect='auto', origin='lower', cmap=cmap, norm=LogNorm(vmin=min_val, vmax=max_val))
    ax[1].set_ylabel("Wavenumber (cycles)", fontsize=7)
    ax[1].set_xticks(datetime_monthstart_idxs)
    ax[1].text(0.99, 0.95, "Ground Truth", fontsize=14, color='w', verticalalignment='top', horizontalalignment='right', transform=ax[1].transAxes)

    plot2 = ax[2].imshow(10*np.log10(spectra_realizations_all_numpy[1:]) - 10*np.log10(spectra_groundtruth_all_numpy[1:]), aspect='auto', origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
    ax[2].set_ylabel("Wavenumber (cycles)", fontsize=7)
    ax[2].set_xticks(datetime_monthstart_idxs)
    if daily_max:
        ax[2].set_xticklabels([experiment.timestamps[i*24][:-3] for i in datetime_monthstart_idxs], rotation=30, fontsize=6)
    else:
        ax[2].set_xticklabels([experiment.timestamps[i][:-3] for i in datetime_monthstart_idxs], rotation=30, fontsize=6)

    ylabels = np.append([0], np.arange(0, spectra_realizations_all_numpy.shape[0], 8) + 6, axis=0)
    for a in ax:
        a.set_yticks(ylabels)
        a.set_yticklabels(ylabels+1, fontsize=6)

    cbar0 = fig.colorbar(plot0, ax=ax[:2], orientation='vertical', label='Power Density (km)', pad=0.003, shrink=0.995, aspect=39)
    cbar0.ax.tick_params(labelsize=6, width=.25)

    cbar1 = fig.colorbar(plot2, ax=ax[2], orientation='vertical', label='Power Density Difference (dB)', pad=0.003, shrink=0.995)
    cbar1.ax.tick_params(labelsize=6, width=.25)

    if daily_max:
        _save_figure(fig, f"{var}_dailymax_spectrogram", experiment)
    else:
        _save_figure(fig, f"{var}_spectrogram", experiment)
