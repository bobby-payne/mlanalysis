import torch
import numpy as np


def get_power_spectrum(x):
    x_fft = torch.fft.fftn(x, norm='ortho')
    energy = torch.abs(x_fft).pow(2)
    return energy


def get_wave_number_radial(dim, d=1.0):
    freq = torch.fft.fftfreq(dim, d=d)
    grid_h, grid_w = torch.meshgrid(freq, freq, indexing="ij")  # must be 'ij'
    wave_radial = torch.sqrt(grid_h.pow(2) + grid_w.pow(2))
    return wave_radial


def get_rapsd(x, d=1.0):
    """
    Compute the radially averaged power spectral density (RAPSD) of a 2D square field.

    The RAPSD describes how the variance (energy) of a spatial field is distributed
    across spatial frequencies (wavenumbers), averaged over all directions. It is
    particularly useful in analyzing the spatial structure or scaling behavior of
    geophysical, meteorological, or image data.

    Parameters
    ----------
    x : torch.Tensor
        A 2D square tensor representing the spatial field (e.g., an image or gridded data).
        Must have shape (N, N).
    d : float, optional
        The grid spacing or physical distance between adjacent grid points.
        Defaults to 1.0.

    Returns
    -------
    bin_avgs : torch.Tensor
        The average power spectral density within each radial wavenumber bin.
    bins_mids : torch.Tensor
        The midpoints of the radial wavenumber bins.
    bin_counts : torch.Tensor
        The number of Fourier components contributing to each radial bin.

    Notes
    -----
    - The function computes the 2D Fourier transform of the input, converts it
      to an energy spectrum (squared magnitude), and bins it radially based on
      isotropic wavenumber magnitude.
    - The result provides a one-dimensional representation of the power spectrum,
      where `bins_mids` correspond to the isotropic wavenumber and `bin_avgs`
      gives the corresponding average spectral power.
    """

    if not (x.dim() == 2 and x.shape[0] == x.shape[1]):
        raise ValueError("Input x must be a square 2D tensor")

    dim = x.shape[0]
    wavenumber = wave_number_radial(dim, d=d)

    delta = wavenumber[0][1]
    freq_max = 1 / (2 * d)

    bins_edges = torch.arange(delta / 2, freq_max + delta / 2, delta)
    bins_edges = torch.cat((torch.tensor([0.0]), bins_edges))
    bins_mids = 0.5 * (bins_edges[1:] + bins_edges[:-1])
    bin_counts, _ = torch.histogram(wavenumber, bins=bins_edges)

    energy = power_spectrum(x)
    bin_sums, _ = torch.histogram(wavenumber, bins=bins_edges, weight=energy)
    bin_avgs = bin_sums / bin_counts

    return bin_avgs, bins_mids, bin_counts


def get_log_spectral_distance(ps, ps_ref):
    """Compute the log-spectral distance between two power spectra."""
    lsd = 10*torch.sqrt(torch.mean((torch.log(ps) - torch.log(ps_ref)).pow(2)))
    return lsd


def get_log_spectral_bias(ps, ps_ref):
    """Compute the log-spectral bias between two power spectra."""
    lsb = 10*torch.mean(torch.log(ps) - torch.log(ps_ref))
    return lsb
