# -----------------------------------------------------------------------------
# Name:        fft.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
"""A set of Magnetic Data routines."""

import numpy as np
from scipy.fft import fft2, fftshift, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.stats import binned_statistic


def fftprep(data):
    """
    FFT preparation.

    This routine pads using minimum curvature gridding.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        Input dataset.

    Returns
    -------
    zfin : numpy array.
        Output prepared data.
    datamedian : float
        Median of data.

    """
    datamedian = np.ma.median(data.data)

    nr, nc = data.data.shape
    xmin, xmax, ymin, ymax = data.extent

    nmax = np.max([nr, nc])
    npts = int(2**nextpow2(nmax))

    cdiff = int(np.floor((npts - nc) / 2))
    rdiff = int(np.floor((npts - nr) / 2))
    cdiff2 = npts - cdiff - nc
    rdiff2 = npts - rdiff - nr

    zfin = data.copy()
    zfin.data = zfin.data - datamedian
    zfin.data = zfin.data.filled(0)
    nr, nc = zfin.data.shape

    zfin.data = np.pad(zfin.data, [[rdiff, rdiff2], [cdiff, cdiff2]],
                       mode='constant', constant_values=0)

    dx = zfin.xdim
    dy = zfin.ydim

    xmin2 = xmin - cdiff * dx
    ymax2 = ymax + rdiff2 * dy

    zfin.set_transform(xdim=dx, xmin=xmin2, ydim=dy, ymax=ymax2)

    return zfin, datamedian


def fft_getkxy(fftmod, xdim, ydim):
    """
    Get KX and KY.

    Parameters
    ----------
    fftmod : numpy array
        FFT data.
    xdim : float
        cell x dimension.
    ydim : float
        cell y dimension.

    Returns
    -------
    KX : numpy array
        x sample frequencies.
    KY : numpy array
        y sample frequencies.

    """
    ny, nx = fftmod.shape
    kx = fftfreq(nx, xdim) * 2 * np.pi
    ky = fftfreq(ny, ydim) * 2 * np.pi

    KX, KY = np.meshgrid(kx, ky)
    KY = -KY
    return KX, KY


def nextpow2(n):
    """
    Next power of 2.

    Parameters
    ----------
    n : float or numpy array
        Current value.

    Returns
    -------
    m_i : float or numpy array
        Output.

    """
    m_i = np.ceil(np.log2(np.abs(n)))
    return m_i


def calculate_raps(dat):
    """
    Calculates the Radially Averaged Power Spectrum (RAPS) of a 2D dataset.

    Parameters
    ----------
    dat : np.ndarray
        A 2D NumPy array of the geophysical data.

    Returns
    -------
    k : np.ndarray
        The 1D array of radial wavenumbers.
    raps : np.ndarray
        The 1D array of radially averaged power spectrum values.
    """
    data = dat.data
    dx = dat.xdim
    dy = dat.ydim

    # 1. Take the 2D FFT of the input data.
    # The output is a complex array.
    F = fft2(data)

    # 2. Shift the zero-frequency component to the center.
    F_shifted = fftshift(F)

    # 3. Calculate the 2D power spectrum.
    power_spectrum_2D = np.abs(F_shifted)**2

    # 4. Create 2D arrays of frequency coordinates.
    ny, nx = data.shape
    kx = 2 * np.pi * fftshift(fftfreq(nx, d=dx))
    ky = 2 * np.pi * fftshift(fftfreq(ny, d=dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # 5. Calculate the radial wavenumber (k) for each point.
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)

    # Bin the power spectrum by radial wavenumber.
    k_bins = np.linspace(k_radial.min(), k_radial.max(), num=100)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Use scipy's binned_statistic to perform the radial average.
    raps, _, _ = binned_statistic(
        k_radial.ravel(),
        power_spectrum_2D.ravel(),
        statistic='mean',
        bins=k_bins)

    nyq = np.pi / min(dx, dy)

    raps = raps[k_centers < nyq]
    k_centers = k_centers[k_centers < nyq]

    k_radial = np.fft.fftshift(k_radial)

    # remove nan  bins
    filt = ~np.isnan(raps)
    k_centers = k_centers[filt]
    raps = raps[filt]

    return k_centers, raps, k_radial, F


def _testfft():
    """Test FFT."""
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster

    ifile = r"c:\workdata\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"

    data = get_raster(ifile)[0]

    datm, _ = fftprep(data)

    plt.figure()
    plt.title('datm')
    vmin, vmax = datm.get_vmin_vmax()
    plt.imshow(datm.data, vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.show()

    xm, ym, _, _ = calculate_raps(datm)

    plt.figure()
    plt.title('datm')
    plt.semilogy(xm, ym, 'b')
    plt.tight_layout()
    plt.show()


def _testfft1d():
    """Test FFT."""
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster
    # import pycurious

    ifile = r"c:\workdata\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"
    ifile = r"D:\Heliium_Highresmag_utm35s.hdr"
    ifile = r"D:\heliumtest.tif"

    data = get_raster(ifile)[0]
    dt = data.xdim

    datm, _ = fftprep(data)

    plt.figure()
    plt.title('datm')

    for data in datm.data:
        if data.max() == 0:
            continue

        N = len(data)  # Number of samples
        yf = rfft(data)
        xm = rfftfreq(N, dt)

        ym = np.abs(yf)**2 / N

        plt.semilogy(xm, ym)

    for data in datm.data.T:
        if data.max() == 0:
            continue

        N = len(data)  # Number of samples
        yf = rfft(data)
        xm = rfftfreq(N, dt)

        ym = np.abs(yf)**2 / N

        plt.semilogy(xm, ym)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _testfft()
