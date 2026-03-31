import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from .utils import compute_daily_maximum, compute_statistics, compute_ranks
from .spectral import get_rapsd, compute_realizations_spectra


def _save_figure(fig, filename, experiment, save_to_experiment_root=False):

    path_to_output = experiment.path_to_output
    experiment_name = experiment.experiment_name
    year = experiment.year
    format = experiment.output_fig_format
    dpi = experiment.output_fig_dpi
    if save_to_experiment_root:
        path_to_output_full = os.path.join(
            path_to_output,
            experiment_name,
            f"{filename}.{format}"
        )
    else:
        path_to_output_full = os.path.join(
            path_to_output,
            experiment_name,
            str(year),
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
        _save_figure(plt.gcf(), f"training_{m}", experiment, save_to_experiment_root=True)

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
        realizations,
        d=4.0,
    )

    # ground truth spectrum
    groundtruth_data = experiment.data_scaled['groundtruth'][var][time_idx].squeeze()
    groundtruth_spectrum, _, _ = get_rapsd(groundtruth_data, d=4.0)

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
    groundtruth_timeseries = torch.where(groundtruth_timeseries == 0, np.nan, groundtruth_timeseries) # fire season inactive
    first_active_idx = torch.where(~torch.isnan(groundtruth_timeseries))[0][0]
    last_active_idx = torch.where(~torch.isnan(groundtruth_timeseries))[0][-1]
    if last_active_idx == first_active_idx:
        last_active_idx = len(experiment.timestamps)-1
    sf = experiment.scale_factor
    try:
        covariate_timeseries = experiment.data_scaled['covariates'][var][:, :, :, y//sf, x//sf].squeeze()
    except KeyError:
        try:
            covariate_timeseries = experiment.data_scaled['covariates'][var + '_c'][:, :, :, y//sf, x//sf].squeeze()
        except KeyError:
            covariate_timeseries = None
    covariate_timeseries = torch.where(covariate_timeseries == 0, np.nan, covariate_timeseries) # fire season inactive

    # ranks for rank histogram
    ranks = compute_ranks(realizations_timeseries, groundtruth_timeseries, time_dim=0)

    # Plot time series and rank histogram
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[.9, .9], hspace=0.25, wspace=0.15)

    ax_top = fig.add_subplot(gs[0, :])   # long plot on top
    ax_bl = fig.add_subplot(gs[1, 0])    # bottom-left
    ax_br = fig.add_subplot(gs[1, 1])    # bottom-right

    # --- top plot: time series ---
    for i in range(N):
        label = f"Realizations (N={N})" if i == 0 else None
        ax_top.plot(realizations_timeseries[:, i].numpy(), color="royalblue", lw=0.6, alpha=0.25, label=label)
    ax_top.plot(torch.mean(realizations_timeseries, dim=1).numpy(), color="navy", lw=1.0, ls="--", label="Realizations Mean")
    ax_top.plot(groundtruth_timeseries.numpy(), color="k", lw=0.9, label="Ground Truth")
    if covariate_timeseries is not None:
        ax_top.plot(covariate_timeseries.numpy(), color="k", lw=0.8, ls=":", label="LR Conditioning")
    ax_top.set_ylabel(var.upper())
    xtick_idxs = [0, len(experiment.timestamps)//2, len(experiment.timestamps)-1]
    ax_top.set_xlim(first_active_idx-2, last_active_idx+2)
    ax_top.set_xticks(xtick_idxs)
    ax_top.set_xticklabels([experiment.timestamps[i] for i in xtick_idxs])
    ax_top.legend(frameon=False, fontsize=8, ncols=2)

    # --- bottom-left: normalized rank histogram ---
    norm_ranks = (ranks / ranks.max()).numpy()
    counts, bins, _ = ax_bl.hist(
        norm_ranks, bins=np.linspace(0, 1, 11), density=True,
        color="0.75", edgecolor="0.25", rwidth=0.9, linewidth=0.6, zorder=10
    )
    ax_bl.axhline(1, color="r", linestyle="--", zorder=11)
    ax_bl.set_xlabel("Normalized Rank")
    ax_bl.set_ylabel("Density")

    # --- bottom-right: rank CDF ---
    ax_br.plot(bins, np.append(0, np.cumsum(counts) * (bins[1:] - bins[:-1])), color="k", label="Empirical")
    ax_br.plot([0, 1], [0, 1], color="r", linestyle="--", label="Ideal")
    ax_br.set_xlabel("Normalized Rank")
    ax_br.set_ylabel("CDF")
    # ax_br.legend(loc="lower right", frameon=False, fontsize=8)

    for a in (ax_top, ax_bl, ax_br):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    dtxt = 0.1
    ax_top.text(-dtxt/2, 1.05, "(a)", ha='left', va='center', fontsize=15, transform=ax_top.transAxes, fontweight='bold')
    ax_bl.text(-dtxt, 1.05, "(b)", ha='left', va='center', fontsize=15, transform=ax_bl.transAxes, fontweight='bold')
    ax_br.text(-dtxt, 1.05, "(c)", ha='left', va='center', fontsize=15, transform=ax_br.transAxes, fontweight='bold')

    _save_figure(fig, f"{var}_timeseries_x{x}y{y}", experiment)


def plot_dailymax_timeseries(experiment, var, N, xy):

    # generate dailymax timeseries
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
    groundtruth_timeseries = torch.where(groundtruth_timeseries == 0, np.nan, groundtruth_timeseries) # fire season inactive
    first_active_idx = torch.where(~torch.isnan(groundtruth_timeseries))[0][0]
    last_active_idx = torch.where(~torch.isnan(groundtruth_timeseries))[0][-1]
    if last_active_idx == first_active_idx:
        last_active_idx = N_days - 1
    sf = experiment.scale_factor
    try:
        covariate_timeseries = experiment.data_scaled['covariates'][var][:, :, :, y//sf, x//sf].squeeze()
    except KeyError:
        try:
            covariate_timeseries = experiment.data_scaled['covariates'][var + '_c'][:, :, :, y//sf, x//sf].squeeze()
        except KeyError:
            covariate_timeseries = None
    if covariate_timeseries is not None:
        covariate_timeseries = compute_daily_maximum(covariate_timeseries, axis=0)
    covariate_timeseries = torch.where(covariate_timeseries == 0, np.nan, covariate_timeseries) # fire season inactive
    N_days = realizations_timeseries.shape[0]

    # ranks for rank histogram of daily maxima
    ranks = compute_ranks(realizations_timeseries, groundtruth_timeseries, time_dim=0)

    # Plot time series and rank histogram
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[.9, .9], hspace=0.25, wspace=0.15)

    ax_top = fig.add_subplot(gs[0, :])   # long plot on top
    ax_bl = fig.add_subplot(gs[1, 0])    # bottom-left
    ax_br = fig.add_subplot(gs[1, 1])    # bottom-right

    # --- top plot: time series ---
    for i in range(N):
        label = f"Realizations (N={N})" if i == 0 else None
        ax_top.plot(realizations_timeseries[:, i].numpy(), color="royalblue", lw=0.6, alpha=0.25, label=label)
    ax_top.plot(torch.mean(realizations_timeseries, dim=1).numpy(), color="navy", lw=1.0, ls="--", label="Realizations Mean")
    ax_top.plot(groundtruth_timeseries.numpy(), color="k", lw=0.9, label="Ground Truth")
    if covariate_timeseries is not None:
        ax_top.plot(covariate_timeseries.numpy(), color="k", lw=0.8, ls=":", label="LR Conditioning")
    ax_top.set_ylabel(var.upper())
    xtick_idxs = [0, N_days//2, N_days-1]
    ax_top.set_xticks(xtick_idxs)
    ax_top.set_xticklabels([experiment.timestamps[i * 24][:10] for i in xtick_idxs])
    ax_top.set_xlim(first_active_idx, last_active_idx)
    ax_top.legend(frameon=False, fontsize=8, ncols=2)

    # --- bottom-left: normalized rank histogram ---
    norm_ranks = (ranks / ranks.max()).numpy()
    counts, bins, _ = ax_bl.hist(
        norm_ranks, bins=np.linspace(0, 1, 11), density=True,
        color="0.75", edgecolor="0.25", rwidth=0.9, linewidth=0.6, zorder=10
    )
    ax_bl.axhline(1, color="r", linestyle="--", zorder=11)
    ax_bl.set_xlabel("Normalized Rank")
    ax_bl.set_ylabel("Density")

    # --- bottom-right: rank CDF ---
    ax_br.plot(bins, np.append(0, np.cumsum(counts) * (bins[1:] - bins[:-1])), color="k", label="Empirical")
    ax_br.plot([0, 1], [0, 1], color="r", linestyle="--", label="Ideal")
    ax_br.set_xlabel("Normalized Rank")
    ax_br.set_ylabel("CDF")
    # ax_br.legend(loc="lower right", frameon=False, fontsize=8)

    for a in (ax_top, ax_bl, ax_br):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    dtxt = 0.1
    ax_top.text(-dtxt/2, 1.05, "(a)", ha='left', va='center', fontsize=15, transform=ax_top.transAxes, fontweight='bold')
    ax_bl.text(-dtxt, 1.05, "(b)", ha='left', va='center', fontsize=15, transform=ax_bl.transAxes, fontweight='bold')
    ax_br.text(-dtxt, 1.05, "(c)", ha='left', va='center', fontsize=15, transform=ax_br.transAxes, fontweight='bold')

    _save_figure(fig, f"{var}_dailymax_timeseries_x{x}y{y}", experiment)


def plot_pixelwise_statistics(experiment, var, N, daily_max=False):

    # ------------------
    # Data preparation and statistics computation
    # ------------------

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
        median_field_sr,
        std_field_sr,
        iqr_field_sr,
        p95_field_sr,
        p99_field_sr) = compute_statistics(realizations_timeseries, prestacked=True, axis=0)

    (mean_field_gt,
        median_field_gt,
        std_field_gt,
        iqr_field_gt,
        p95_field_gt,
        p99_field_gt) = compute_statistics(groundtruth_timeseries, prestacked=True, axis=0)

    # Average over realizations and plot statistics
    stats = {
        'Mean': (mean_field_sr, mean_field_gt),
        'Median': (median_field_sr, median_field_gt),
        'Standard Deviation': (std_field_sr, std_field_gt),
        'Interquartile Range': (iqr_field_sr, iqr_field_gt),
        '95 Percentile': (p95_field_sr, p95_field_gt),
        '99 Percentile': (p99_field_sr, p99_field_gt)
    }
    stats = {statname: (np.nanmean(fields[0], axis=0), fields[1]) for statname, fields in stats.items()}

    # ===============================
    # Figure layout
    # ===============================

    n_stats = len(stats)

    fig = plt.figure(figsize=(10, n_stats * 3 + 1))

    # Outer grid: n_stats rows, 2 columns (maps | histogram)
    gs = GridSpec(
        n_stats,
        2,
        figure=fig,
        width_ratios=[3, 1],   # maps take 3x the width of histogram
        hspace=0.4,
        wspace=0.15,
    )

    # ===============================
    # Spatial statistics panels
    # ===============================

    cmap = 'viridis'
    ts = 8
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    ax_hist_list = []

    for idx, (stat_name, stat_fields) in enumerate(stats.items()):

        # ---- Maps (left cell) ----
        ax_maps = fig.add_subplot(gs[idx, 0])

        vmin = min(np.nanmin(stat_fields[0]), np.nanmin(stat_fields[1])).item()
        vmax = max(np.nanmax(stat_fields[0]), np.nanmax(stat_fields[1])).item()
        dvmin = np.nanmin(stat_fields[0] - stat_fields[1]).item()
        dvmax = np.nanmax(stat_fields[0] - stat_fields[1]).item()

        if np.abs(dvmin) < np.abs(dvmax):
            dvmin = -dvmax
        elif np.abs(dvmin) > np.abs(dvmax):
            dvmax = -dvmin

        # Inner grid: 1 row, 3 maps
        gs_inner = ax_maps.get_subplotspec().subgridspec(1, 3,)
        ax_maps.remove()

        ax_left  = fig.add_subplot(gs_inner[0])
        ax_mid   = fig.add_subplot(gs_inner[1])
        ax_right = fig.add_subplot(gs_inner[2])

        for axis in [ax_left, ax_mid, ax_right]:
            axis.axis('off')

        ax_left.set_title("Downscaled", fontsize=ts)
        ax_left.imshow(stat_fields[0], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        ax_mid.set_title("Ground Truth", fontsize=ts)
        plot1 = ax_mid.imshow(stat_fields[1], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

        ax_right.set_title("Downscaled - Ground Truth", fontsize=ts)
        plot2 = ax_right.imshow(stat_fields[0] - stat_fields[1],
                                cmap='RdBu_r', vmin=dvmin, vmax=dvmax,
                                origin='lower')

        # Row title + letter label
        x0 = ax_left.get_position().x0
        x1 = ax_right.get_position().x1
        y1 = max(ax_left.get_position().y1,
                 ax_mid.get_position().y1,
                 ax_right.get_position().y1)

        display_name = stat_name
        if stat_name.endswith("Percentile"):
            display_name = stat_name.replace(" ", "th ")

        dy = 0.016
        fig.text((x0 + x1) / 2, y1 + dy,
                 display_name,
                 ha='center', va='bottom',
                 fontsize=ts + 2, fontweight='bold')

        fig.text(x0, y1 + dy,
                 letters[idx],
                 ha='center', va='bottom',
                 fontsize=ts + 2, fontweight='bold')

        # Colorbars
        colorbar = fig.colorbar(plot1,
                                ax=[ax_left, ax_mid],
                                orientation='horizontal',
                                pad=0.05, fraction=0.0512, aspect=40)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=ts, width=0.25)
        colorbar.set_label(var.upper(), fontsize=ts)

        colorbar_dif = fig.colorbar(plot2,
                                    ax=ax_right,
                                    orientation='horizontal',
                                    pad=0.05, fraction=0.045, aspect=20)
        colorbar_dif.outline.set_visible(False)
        colorbar_dif.ax.tick_params(labelsize=ts, width=0.25)
        colorbar_dif.set_label("$\Delta$" + var.upper(), fontsize=ts)

        # ---- Histogram (right cell) ----
        ax_h = fig.add_subplot(gs[idx, 1])
        ax_hist_list.append((ax_h, stat_name))

    # ===============================
    # Histogram panels
    # ===============================

    nbins = 200

    for i, (axis, statname) in enumerate(ax_hist_list):

        min_val = np.nanmin([stats[statname][0].min(), stats[statname][1].min()])
        max_val = np.nanmax([stats[statname][0].max(), stats[statname][1].max()])
        bins = np.linspace(min_val, max_val, nbins)

        axis.hist(stats[statname][1].flatten(),
                bins=bins, alpha=0.52, label='Ground Truth', density=0)

        axis.hist(stats[statname][0].flatten(),
                bins=bins, alpha=0.52, label='Downscaled',
                color='orange', density=0)

        legend_kwargs = dict(fontsize=8, frameon=False,
                            loc='upper right' if i == 0 else 'best')
        axis.legend(**legend_kwargs)

        axis.tick_params(axis='both', which='major', labelsize=7)
        axis.set_xlabel(var.upper(), size=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(y=0.05)
        pos = axis.get_position()
        axis.set_position([pos.x0, pos.y0 + 0.003, pos.width * 1.6, pos.height * .95])

    # =======================================
    # Save figure
    # =======================================
    if daily_max:
        _save_figure(fig, f"{var}_dailymax_pixelwise_stats", experiment)
    else:
        _save_figure(fig, f"{var}_pixelwise_stats", experiment)


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
