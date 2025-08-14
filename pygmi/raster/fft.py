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

from pygmi.vector.dataprep import gridxyz
from pygmi.raster.misc import lstack


def fftprepminc(data, showlog=print, piter=iter):
    """
    FFT preparation.

    This routine pads using minimum curvature gridding.

    Parameters
    ----------
    data : pygmi.raster.datatypes.Data
        Input dataset.
    showlog : function, optional
        Show information using a function. The default is print.

    Returns
    -------
    zfin : numpy array.
        Output prepared data.
    datamedian : float
        Median of data.

    """
    datamedian = np.ma.median(data.data)
    ndat = data.data - datamedian

    nr, nc = data.data.shape
    cdiff = nc // 2
    rdiff = nr // 2

    xmin, xmax, ymin, ymax = data.extent

    x = np.arange(xmin, xmax, data.xdim) + data.xdim / 2
    y = np.arange(ymin, ymax, data.ydim) + data.ydim / 2
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    x, y = np.meshgrid(x, y)
    z = ndat
    y = y[::-1]
    x = np.ma.array(x, mask=z.mask)
    y = np.ma.array(y, mask=z.mask)

    x = x.compressed()
    y = y.compressed()
    z = z.compressed()
    dxy = min(data.xdim, data.ydim)
    xmin2, xmax2 = [xmin - cdiff * dxy, xmax + cdiff * dxy]
    ymin2, ymax2 = [ymin - rdiff * dxy, ymax + rdiff * dxy]

    x2 = np.arange(xmin2, xmax2, dxy).tolist()
    y2 = np.arange(ymin2, ymax2, dxy).tolist()

    xcnr = x2 * 2 + [xmin2] * len(y2) + [xmax2] * len(y2)
    ycnr = [ymin2] * len(x2) + [ymax2] * len(x2) + y2 * 2
    zcnr = np.zeros_like(xcnr)

    x = np.append(x, xcnr)
    y = np.append(y, ycnr)
    z = np.append(z, zcnr)

    zfin = gridxyz(x, y, z, dxy, method='Minimum Curvature', bdist=None,
                   showlog=showlog)

    zfin.data[np.isnan(zfin.data)] = 0.
    zfin.crs = data.crs

    tmp = lstack([zfin, data], showlog=showlog, piter=piter)
    tmp2 = tmp[1]
    tmp2.data = tmp2.data - datamedian
    tmp2.data[tmp2.data.mask] = tmp[0].data[tmp2.data.mask]
    zfin = tmp2

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
    kx = np.fft.fftfreq(nx, xdim) * 2 * np.pi
    ky = np.fft.fftfreq(ny, ydim) * 2 * np.pi

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


def radial_average_power_spectrum(dat):
    """
    Calculate the radially averaged power spectrum.

    Parameters
    ----------
    data : PyGMI data
        Input data.

    Returns
    -------
    radial_bins : numpy array
        1D radial wavenumbers.
    radial_mean : numpy array
        1D radial power spectrum.
    freq_radius : numpy array
        2D wavenumber array.
    fft_data : numpy array
        2D FFT data array.

    """
    data = dat.data
    dx = dat.xdim
    dy = dat.ydim
    # Compute the 2D Fourier Transform
    fft_data = np.fft.fft2(data)
    # fft_shifted = np.fft.fftshift(fft_data)
    power_spectrum = np.abs(fft_data) ** 2

    # Get the frequency coordinates
    ny, nx = data.shape
    fx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    fy = np.fft.fftfreq(ny, dy) * 2 * np.pi
    fx, fy = np.meshgrid(fx, fy)
    freq_radius = np.sqrt(fx**2 + fy**2)

    # Shift the frequency coordinates to match the shifted FFT
    # freq_radius = np.fft.fftshift(freq_radius)

    # Radial binning
    max_radius = (np.max(freq_radius))
    radial_bins = np.linspace(0, max_radius, min(nx, ny))
    radial_mean = np.zeros_like(radial_bins, dtype=float)
    radial_indices = np.digitize(freq_radius.ravel(), radial_bins)

    radial_mean = []
    for i in range(1, len(radial_bins)):
        mask = radial_indices == i
        radial_mean.append(np.mean(power_spectrum.ravel()[mask]))
        pass

    # Compute bin centers
    radial_bins = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    mask = ~np.isnan(radial_mean)
    radial_bins = radial_bins[mask]
    radial_mean = np.array(radial_mean)[mask]

    return radial_bins, radial_mean, freq_radius, fft_data


def _testfft():
    """Test FFT."""
    import matplotlib.pyplot as plt
    from pygmi.raster.iodefs import get_raster

    ifile = r"c:\workdata\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"

    data = get_raster(ifile)[0]
    datm, _ = fftprepminc(data)

    plt.figure()
    plt.title('datm')
    plt.imshow(datm.data)

    plt.tight_layout()
    plt.show()

    xm, ym, _, _ = radial_average_power_spectrum(datm)

    plt.figure()
    plt.title('datm')
    plt.semilogy(xm, ym, 'b')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _testfft()
