# -----------------------------------------------------------------------------
# Name:        transforms.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2021 Council for Geoscience
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
Transforms such as PCA and MNF.
"""
import sys
import os
import glob
import copy

import numpy as np
from PyQt5 import QtWidgets, QtCore
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from osgeo import gdal
import spectral as sp

from pygmi.raster.iodefs import get_raster
from pygmi.misc import ProgressBarText
from pygmi.raster.iodefs import export_gdal
from pygmi.misc import PTime


def main_pca():
    """PCA."""

    idir = r'C:\Work\Workdata\Richtersveld\Reprocessed'
    odir = r'C:\Work\Workdata\Richtersveld\PCA'
    cmps = 7
    windows = [[0, 322],
               [0, 72],
               [72, 172],
               [172, 210],
               [210, 262],
               [262, 322]]
    win = windows[-1]

    allfiles = glob.glob(os.path.join(idir, '*.dat'))
    allfiles = [allfiles[0]]

    pbar = ProgressBarText()

    for ifile in allfiles:
        ofile = os.path.join(odir, os.path.basename(ifile[:-4]) +
                             f'{win[0]}_{win[1]}_I_PCA.tif')

        print('Importing data...')
        print(os.path.basename(ifile))
        dataset = gdal.Open(ifile, gdal.GA_ReadOnly)
        rowsall = dataset.RasterYSize
        dataset = None
        rinc = 10

        print('Fitting PCA')
        ttt = PTime()
        pca = IncrementalPCA(n_components=cmps)

        pca.n_samples_seen_ = np.int64(0)
        pca.mean_ = .0
        pca.var_ = .0

        for i in pbar.iter(range(0, rowsall, rinc)):

            xoff = 0
            yoff = i
            xsize = None
            ysize = rinc
            iraster = (xoff, yoff, xsize, ysize)
            dat = get_raster(ifile, nval=0, iraster=iraster)

            dat2 = []
            for j in dat:
                dat2.append(dat[j].data)
                mask = dat[j].data.mask
                datorig = dat[j]

            dat2 = np.array(dat2)
            dat2 = np.moveaxis(dat2, 0, -1)
            dat2 = dat2[:, :, win[0]:win[1]]
            dat3 = dat2[~mask]

            if dat3.size == 0:
                continue

            if dat3.shape[0] < cmps:
                continue
            pca.partial_fit(dat3)

        ttt.since_last_call('\nFit time')

        np.set_printoptions(suppress=True, precision=3)
        print('Percentage of variance explained by each of the components:')
        print(pca.explained_variance_ratio_*100)

        print('Calculating PCA')
        xpca = []
        maskall = []
        for i in pbar.iter(range(0, rowsall, rinc)):
            xoff = 0
            yoff = i
            xsize = None
            ysize = rinc
            iraster = (xoff, yoff, xsize, ysize)
            dat = get_raster(ifile, nval=0, iraster=iraster)

            dat2 = []
            for j in dat:
                dat2.append(dat[j].data)
                mask = dat[j].data.mask
                datorig = dat[j]

            dat2 = np.array(dat2)
            dat2 = np.moveaxis(dat2, 0, -1)
            dat2 = dat2[:, :, win[0]:win[1]]
            dat3 = dat2[~mask]

            if dat3.size == 0:
                continue

            if dat3.shape[0] < cmps:
                continue

            maskall.append(mask)
            xpca.append(pca.transform(dat3))

        maskall = np.vstack(maskall)
        xpca = np.vstack(xpca)

        ttt.since_last_call('\nTransform time')

        rows, cols = maskall.shape
        datall = np.zeros([rows, cols, cmps])
        datall[~maskall] = xpca

        datfin = {}
        for i in range(cmps):
            rband = copy.copy(datorig)
            rband.data = datall[:, :, i]
            rband.dataid = str(i+1)

            datfin[str(i+1)] = rband

        if datfin:
            export_gdal(ofile, datfin, 'GTiff')

        del datall
        del maskall
        del dat3
        del rband
        del datfin


def mnf_calc(x2d, maskall, ncmps=7):
    """MNF Filtering"""

    dim = x2d.shape[-1]
    mask = maskall[:, :, 0]
    x = x2d[~mask]

    # Diagonal or SPy
    noise = x2d[:-1, :-1] - x2d[1:, 1:]
    mask2 = np.logical_or(mask[:-1, :-1], mask[1:, 1:])
    noise = noise[~mask2]
    ncov = np.cov(noise.T)/2

    # ENVI
    # vdiff = x2d[:-1] - x2d[1:]
    # hdiff = x2d[:, :-1] - x2d[:, 1:]
    # noise = (vdiff[:, :-1]+hdiff[:-1])/2
    # mask2 = np.logical_or(mask[:-1, :-1], mask[1:, 1:])
    # noise = noise[~mask2]
    # ncov = np.cov(noise.T)

    # Calculate evecs and evals

    nevals, nevecs = np.linalg.eig(ncov)

    Ln = np.power(nevals, -0.5)
    Ln = np.diag(Ln)

    W = Ln @ nevecs.T
    Winv = np.linalg.inv(W)

    Pnorm = W @ x.T

    pca = PCA(n_components=ncmps)
    P = pca.fit_transform(Pnorm.T)
    P = pca.inverse_transform(P)

    x2 = (Winv @  P.T).T

    rows, cols = mask.shape
    datall = np.zeros([rows, cols, dim])
    datall[~mask] = x2
    datall = np.ma.array(datall, mask=maskall)

    return datall


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['figure.dpi'] = 300

    pbar = ProgressBarText()

    ifile = r'C:\Workdata\lithosphere\Cut-90-0824-.hdr'
    ncmps = 10
    nodata = 0
    iraster = None

    dat = get_raster(ifile, nval=nodata, iraster=iraster, piter=pbar.iter)

    dat2 = []
    maskall = []
    for j in dat:
        dat2.append(j.data[500:1000, 500:1000])
        mask = j.data[500:1000, 500:1000].mask
        maskall.append(mask)

    maskall = np.moveaxis(maskall, 0, -1)
    dat2 = np.moveaxis(dat2, 0, -1)

    ttt = PTime()
    print('Calculating MNF')
    pmnf = mnf_calc(dat2, maskall, ncmps)

    signal = sp.calc_stats(dat2)
    noise = sp.noise_from_diffs(dat2)
    mnfr = sp.mnf(signal, noise)
    denoised = mnfr.denoise(dat2, num=ncmps)

    ttt.since_last_call()

    for i in [0, 5, 10, 13, 14, 15, 20, 25]:
        plt.title('█████████████████Old dat2 band'+str(i))
        plt.imshow(dat2[:, :, i])
        plt.colorbar()
        plt.show()

        plt.title('SPy MNF denoised band'+str(i))
        plt.imshow(denoised[:, :, i])
        plt.colorbar()
        plt.show()

        plt.title('New MNF denoised band'+str(i))
        plt.imshow(pmnf[:, :, i])
        plt.colorbar()
        plt.show()

    return


if __name__ == "__main__":
    _testfn()
