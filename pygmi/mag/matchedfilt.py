# -----------------------------------------------------------------------------
# Name:        matchedfilt.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2025 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""
Quick start routine to start the GUI form of PyGMI.

This routine is used as a convenience function, typically if you do NOT
formally install PyGMI as a library and prefer to run it from within the
default extracted directory structure.
"""
import winsound

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from pygmi.raster.iodefs import get_raster
from pygmi.raster.dataprep import fftprep, fft_getkxy


def main():
    """Start of program."""
    ifile = r"D:\workdata\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"

    dat = get_raster(ifile)

    data = dat[0]

    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, datamedian = fftprep(data)

    # Calculate the radially averaged power spectrum
    radial_bins, radial_mean = radial_average_power_spectrum(ndat, xdim, ydim)

    # Plot the result
    plt.figure()
    plt.plot(radial_bins, radial_mean,
             label="Radially Averaged Power Spectrum")
    plt.xlabel("Wavenumbers")
    plt.ylabel("Power")
    plt.title("Radially Averaged Power Spectrum")
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.show()


def compute_1d_power_spectrum(data):
    # Step 1: Perform 2D Fourier Transform
    fft2_result = np.fft.fft2(data)
    # Shift zero frequency to the center
    fft2_shifted = np.fft.fftshift(fft2_result)

    # Step 2: Compute the power spectrum
    power_spectrum_2d = np.abs(fft2_shifted) ** 2

    # Step 3: Create radial bins for averaging
    ny, nx = data.shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(int)

    # Step 4: Radially average the power spectrum
    radial_profile = np.bincount(
        r.ravel(), weights=power_spectrum_2d.ravel()) / np.bincount(r.ravel())

    return radial_profile


def radial_average_power_spectrum(data, dx=1.0, dy=1.0):
    # Compute the 2D Fourier Transform
    fft_data = np.fft.fft2(data)
    fft_shifted = np.fft.fftshift(fft_data)
    power_spectrum = np.abs(fft_shifted) ** 2

    # Get the frequency coordinates
    ny, nx = data.shape
    fx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    fy = np.fft.fftfreq(ny, dy) * 2 * np.pi
    fx, fy = np.meshgrid(fx, fy)
    freq_radius = np.sqrt(fx**2 + fy**2)

    # Shift the frequency coordinates to match the shifted FFT
    freq_radius = np.fft.fftshift(freq_radius)

    # Radial binning
    max_radius = (np.max(freq_radius))
    radial_bins = np.linspace(0, max_radius, 256)
    radial_mean = np.zeros_like(radial_bins, dtype=float)
    radial_indices = np.digitize(freq_radius.ravel(), radial_bins)

    radial_mean = []
    for i in range(1, len(radial_bins)):
        mask = radial_indices == i
        radial_mean.append(np.mean(power_spectrum.ravel()[mask]))

    # Compute bin centers
    radial_bins = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    return radial_bins, radial_mean


if __name__ == "__main__":
    main()

    print('Finished!')
    winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
