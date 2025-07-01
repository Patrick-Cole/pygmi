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
from scipy.optimize import curve_fit


from pygmi.raster.iodefs import get_raster
from pygmi.raster.dataprep import fftprep
from pygmi.vector.minc import minc


def main():
    """Start of program."""
    ifile = r"c:\work\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"

    dat = get_raster(ifile)

    data = dat[0]

    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, datamedian = fftprepminc(data)
    ndat2, rdiff, cdiff, datamedian = fftprep(data)
    plt.subplot(121)
    plt.imshow(ndat)
    plt.subplot(122)
    plt.imshow(ndat2)
    plt.show()

    # Calculate the radially averaged power spectrum
    radial_bins, radial_mean = radial_average_power_spectrum(ndat, xdim, ydim)
    radial_bins2, radial_mean2 = radial_average_power_spectrum(
        ndat2, xdim, ydim)
    # Plot the result
    plt.figure()
    plt.plot(radial_bins, radial_mean,
             label="Radially Averaged Power Spectrum")
    plt.plot(radial_bins2, radial_mean2,
             label="Radially Averaged Power Spectrum old")
    plt.xlabel("Wavenumbers")
    plt.ylabel("Power")
    plt.title("Radially Averaged Power Spectrum")
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.show()


def fftprepminc(data):
    """
    FFT preparation.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        Input dataset.

    Returns
    -------
    zfin : numpy array.
        Output prepared data.
    rdiff : int
        rows divided by 2.
    cdiff : int
        columns divided by 2.
    datamedian : float
        Median of data.

    """
    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian

    nr, nc = data.data.shape
    cdiff = nc // 2
    rdiff = nr // 2

    xmin, xmax, ymin, ymax = data.extent

    x = np.arange(xmin, xmax, data.xdim)+data.xdim/2
    y = np.arange(ymin, ymax, data.ydim)+data.ydim/2
    x, y = np.meshgrid(x, y)
    z = ndat
    x = np.ma.array(x, mask=z.mask)
    y = np.ma.array(y, mask=z.mask)

    x = x.compressed()
    y = y.compressed()
    z = z.compressed()
    dxy = min(data.xdim, data.ydim)
    xmin2, xmax2 = [xmin-cdiff*dxy, xmax+cdiff*dxy]
    ymin2, ymax2 = [ymin-rdiff*dxy,  ymax+rdiff*dxy]

    x2 = list(np.arange(xmin2, xmax2, dxy))
    y2 = list(np.arange(ymin2, ymax2, dxy))

    xcnr = x2 * 2 + [xmin2]*len(y2) + [xmax2]*len(y2)
    ycnr = [ymin2]*len(x2) + [ymax2]*len(x2) + y2*2
    zcnr = np.zeros_like(xcnr)

    x = np.append(x, xcnr)
    y = np.append(y, ycnr)
    z = np.append(z, zcnr)

    zfin = minc(x, y, z, dxy)

    zfin[np.isnan(zfin)] = 0.
    zfin = zfin[::-1]

    return zfin, rdiff, cdiff, datamedian


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


# Define a piecewise linear function
def piecewise_linear(x, x0, y0, k1, k2):
    """
    x0, y0: The "knot" point where the two linear segments meet.
    k1: Slope of the first segment.
    k2: Slope of the second segment.
    """
    return np.piecewise(
        x,
        [x < x0, x >= x0],
        [lambda x: k1 * (x - x0) + y0, lambda x: k2 * (x - x0) + y0],
    )


def test():

    ifile = r"c:\work\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"

    dat = get_raster(ifile)

    data = dat[0]

    xdim = data.xdim
    ydim = data.ydim

    ndat, rdiff, cdiff, datamedian = fftprepminc(data)
    # Calculate the radially averaged power spectrum
    x_data, y_data = radial_average_power_spectrum(ndat, xdim, ydim)
    y_data = np.log(y_data)

    # Fit the piecewise linear function to the data
    popt, _ = curve_fit(piecewise_linear, x_data, y_data,
                        bounds=[[0, 0, -10000, -10000],
                                [x_data.max(), y_data.max(), 0, 0]])

    # Extract the fitted parameters
    x0, y0, k1, k2 = popt

    # Plot the results
    plt.scatter(x_data, y_data, label="Data", color="blue", s=10)
    plt.plot(x_data, piecewise_linear(x_data, *popt),
             label="Fitted Curve", color="red")
    plt.axvline(x=x0, color="green", linestyle="--",
                label=f"Knot at x={x0:.2f}")
    plt.legend()
    plt.xlabel("Wavenumbers")
    plt.ylabel("Power")
    plt.title("Piecewise Linear Fit")
    plt.show()

    D = -k1/2
    C = np.sqrt(np.exp(y0-k1*x0))

    d = -k2/2
    c = np.sqrt(np.exp(y0-k2*x0))

    k = x_data
    f1 = 1/(1+c/C*np.exp(k*(D-d)))
    f2 = 1 - f1

    plt.plot(k, f1)
    plt.plot(k, f2)
    plt.show()

    pass


if __name__ == "__main__":
    # main()
    test()

    print('Finished!')
    winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
