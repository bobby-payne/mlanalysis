import os
import numpy as np
import torch
import matplotlib.pyplot as plt


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
