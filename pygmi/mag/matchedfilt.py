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
import pwlf

from pygmi.raster.iodefs import get_raster
from pygmi.raster.misc import lstack
from pygmi.vector.dataprep import gridxyz


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

    x = np.arange(xmin, xmax, data.xdim) + data.xdim / 2
    y = np.arange(ymin, ymax, data.ydim) + data.ydim / 2
    x, y = np.meshgrid(x, y)
    z = ndat
    x = np.ma.array(x, mask=z.mask)
    y = np.ma.array(y, mask=z.mask)[::-1]

    x = x.compressed()
    y = y.compressed()
    z = z.compressed()
    dxy = min(data.xdim, data.ydim)
    xmin2, xmax2 = [xmin - cdiff * dxy, xmax + cdiff * dxy]
    ymin2, ymax2 = [ymin - rdiff * dxy, ymax + rdiff * dxy]

    x2 = list(np.arange(xmin2, xmax2, dxy))
    y2 = list(np.arange(ymin2, ymax2, dxy))

    xcnr = x2 * 2 + [xmin2] * len(y2) + [xmax2] * len(y2)
    ycnr = [ymin2] * len(x2) + [ymax2] * len(x2) + y2 * 2
    zcnr = np.zeros_like(xcnr)

    x = np.append(x, xcnr)
    y = np.append(y, ycnr)
    z = np.append(z, zcnr)

    zfin = gridxyz(x, y, z, dxy, method='Minimum Curvature', bdist=None)

    zfin.data[np.isnan(zfin.data)] = 0.

    return zfin, datamedian


def radial_average_power_spectrum(data, dx=1.0, dy=1.0):
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
    radial_bins = np.linspace(0, max_radius, 256)
    radial_mean = np.zeros_like(radial_bins, dtype=float)
    radial_indices = np.digitize(freq_radius.ravel(), radial_bins)

    radial_mean = []
    for i in range(1, len(radial_bins)):
        mask = radial_indices == i
        radial_mean.append(np.mean(power_spectrum.ravel()[mask]))

    # Compute bin centers
    radial_bins = 0.5 * (radial_bins[:-1] + radial_bins[1:])

    return radial_bins, radial_mean, freq_radius, fft_data


def test():

    ifile = r"c:\workdata\PyGMI Test Data\Magnetics\IGRF\MAGMICROLEVEL.ers"
    nsegs = 3
    n = [0, 1, 1]

    dat = get_raster(ifile)

    data = dat[0]

    xdim = data.xdim
    ydim = data.ydim

    data2, datamedian = fftprepminc(data)
    ndat = data2.data
    # Calculate the radially averaged power spectrum
    x_data, y_data, freq_radius, fft_data = radial_average_power_spectrum(
        ndat, xdim, ydim)
    y_data = np.log(y_data)

    my_pwlf = pwlf.PiecewiseLinFit(x_data, y_data)
    breaks = my_pwlf.fit(nsegs)

    x_hat = np.linspace(x_data.min(), x_data.max(), 100)
    y_hat = my_pwlf.predict(x_hat)

    # Plot the results
    plt.scatter(x_data, y_data, label="Data", color="blue", s=10)
    plt.plot(x_hat, y_hat, label="Fitted Curve", color="red")

    # plt.axvline(x=x0, color="green", linestyle="--",
    #             label=f"Knot at x={x0:.2f}")
    plt.legend()
    plt.xlabel("Wavenumbers")
    plt.ylabel("Power")
    plt.title("Piecewise Linear Fit")
    plt.show()

    m = my_pwlf.calc_slopes()
    d = -m / 2
    x0 = breaks[:-1]
    logy0 = my_pwlf.predict(x0)

    c = []
    for i in range(nsegs):
        c.append(
            np.sqrt(np.exp(logy0[i] - n[i] * np.log(x0[i]) - m[i] * x0[i])))

    c = np.array(c)
    k = x_data

    fsum = 0
    for i in range(nsegs):
        fsum += c[i] * k**n[i] * np.exp(-k * d[i])

    f = []
    for i in range(nsegs):
        f.append(c[i] * k**n[i] * np.exp(-k * d[i]) / fsum)

    for i in f:
        plt.plot(k, i)

    plt.plot(k, np.sum(f, 0))
    plt.xlabel("Wavenumbers")
    plt.show()

    k = freq_radius
    fsum = 0
    for i in range(nsegs):
        fsum += c[i] * k**n[i] * np.exp(-k * d[i])

    f = []
    odat = []
    for i in range(nsegs):
        f.append(c[i] * k**n[i] * np.exp(-k * d[i]) / fsum)

        zout = np.real(np.fft.ifft2(fft_data * f[i]))
        zout = zout + datamedian
        tmp = data2.copy()
        tmp.data = np.ma.array(zout)
        tmp.dataid = f'depth {d[i]:.2f}'
        tmp = lstack([tmp, data], masterid=data.dataid, commonmask=True)[0]

        odat.append(tmp)

    plt.subplot(221)
    plt.title(data.dataid)
    vmin, vmax = data.get_vmin_vmax()
    plt.imshow(data.data, vmin=vmin, vmax=vmax, extent=data.extent)
    plt.subplot(222)
    plt.title(odat[0].dataid)
    vmin, vmax = odat[0].get_vmin_vmax()
    plt.imshow(odat[0].data, vmin=vmin, vmax=vmax, extent=odat[0].extent)
    plt.subplot(223)
    plt.title(odat[1].dataid)
    vmin, vmax = odat[1].get_vmin_vmax()
    plt.imshow(odat[1].data, vmin=vmin, vmax=vmax, extent=odat[1].extent)
    plt.subplot(224)
    plt.title(odat[2].dataid)
    vmin, vmax = odat[2].get_vmin_vmax()
    plt.imshow(odat[2].data, vmin=vmin, vmax=vmax, extent=odat[2].extent)

    plt.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    test()

    print('Finished!')
    winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
