import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import get_config


def _save_figure(fig, filename, dpi=150):

    config = get_config()
    path_to_output = config['path_to_output']
    experiment_name = config['experiment_name']
    path_to_output_full = os.path.join(path_to_output, experiment_name, filename)
    os.makedirs(os.path.dirname(path_to_output_full), exist_ok=True)

    fig.savefig(
        path_to_output_full,
        bbox_inches='tight',
        dpi=dpi
    )
    plt.close(fig)


def plot_metrics(metrics_dict, dpi=150):

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
        _save_figure(plt.gcf(), f"training_{m}.png", dpi=dpi)

    plt.close('all')
