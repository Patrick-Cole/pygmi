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
"""Transforms such as PCA and MNF."""

# from memory_profiler import profile
import os
import copy
from math import ceil

import numpy as np
from PyQt5 import QtWidgets, QtCore
from sklearn.decomposition import IncrementalPCA
import numexpr as ne
import matplotlib.pyplot as plt

from pygmi.raster.iodefs import get_raster, export_gdal
from pygmi.misc import ProgressBarText
# from pygmi.raster.iodefs import export_gdal
import pygmi.menu_default as menu_default
# from pygmi.misc import getinfo


class MNF(QtWidgets.QDialog):
    """
    Perform MNF Transform.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
            self.piter = ProgressBarText().iter

        else:
            self.showprocesslog = parent.showprocesslog
            self.piter = parent.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.ev = None

        self.sb_comps = QtWidgets.QSpinBox()
        self.cb_fwdonly = QtWidgets.QCheckBox('Forward Transform Only.')
        self.rb_noise_diag = QtWidgets.QRadioButton('Noise estimated by '
                                                    'diagonal shift')
        self.rb_noise_hv = QtWidgets.QRadioButton('Noise estimated by average '
                                                  'of horizontal and vertical '
                                                  'shift')
        self.rb_noise_quad = QtWidgets.QRadioButton('Noise estimated by local '
                                                    'quadratic surface')

        self.setupui()

        self.resize(500, 350)

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.mnf')
        lbl_comps = QtWidgets.QLabel('Number of components:')

        self.cb_fwdonly.setChecked(True)
        self.sb_comps.setEnabled(False)
        self.sb_comps.setMaximum(10000)
        self.sb_comps.setMinimum(1)
        self.rb_noise_hv.setChecked(True)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Minimum Noise Fraction')

        gridlayout_main.addWidget(self.cb_fwdonly, 1, 0, 1, 2)
        gridlayout_main.addWidget(lbl_comps, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.sb_comps, 2, 1, 1, 1)
        gridlayout_main.addWidget(self.rb_noise_hv, 3, 0, 1, 2)
        gridlayout_main.addWidget(self.rb_noise_diag, 4, 0, 1, 2)
        gridlayout_main.addWidget(self.rb_noise_quad, 5, 0, 1, 2)

        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        self.cb_fwdonly.stateChanged.connect(self.changeoutput)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        self.ev = None
        tmp = []
        if 'Raster' not in self.indata and 'RasterFileList' not in self.indata:
            self.showprocesslog('No Satellite Data')
            return False

        if 'Raster' in self.indata:
            indata = self.indata['Raster']
            self.sb_comps.setMaximum(len(indata))
            self.sb_comps.setValue(ceil(len(indata)*0.04))

        if not nodialog:
            tmp = self.exec_()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

        if not nodialog and self.ev is not None:
            plt.figure('Explained Variance')
            plt.subplot(1, 1, 1)
            plt.plot(self.ev)
            plt.xlabel('Component')
            plt.ylabel('Explained Variance')
            plt.grid(True)
            plt.tight_layout()

            if hasattr(plt.get_current_fig_manager(), 'window'):
                plt.get_current_fig_manager().window.setWindowIcon(self.parent.windowIcon())

            plt.show()

        return True

    def changeoutput(self):
        """
        Change the interface to reflect whether full calculation is needed.

        Returns
        -------
        None.

        """
        uienabled = not self.cb_fwdonly.isChecked()
        self.sb_comps.setEnabled(uienabled)

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        # self.combo_sensor.setCurrentText(projdata['sensor'])
        # self.setratios()

        # for i in self.lw_ratios.selectedItems():
        #     if i.text()[2:] not in projdata['ratios']:
        #         i.setSelected(False)
        # self.set_selected_ratios()

        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}
        # projdata['sensor'] = self.combo_sensor.currentText()

        # rlist = []
        # for i in self.lw_ratios.selectedItems():
        #     rlist.append(i.text()[2:])

        # projdata['ratios'] = rlist

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        ncmps = self.sb_comps.value()
        odata = []
        if self.cb_fwdonly.isChecked():
            ncmps = None

        if self.rb_noise_diag.isChecked():
            noise = 'diagonal'
        elif self.rb_noise_hv.isChecked():
            noise = 'hv average'
        else:
            noise = 'quad'

        if 'RasterFileList' in self.indata:
            flist = self.indata['RasterFileList']
            odir = os.path.join(os.path.dirname(flist[0]), 'feature')

            os.makedirs(odir, exist_ok=True)
            for ifile in flist:
                self.showprocesslog('Processing '+os.path.basename(ifile))

                dat = get_raster(ifile)
                odata, self.ev = mnf_calc(dat, ncmps, piter=self.piter,
                                          pprint=self.showprocesslog,
                                          noisetxt=noise)

                ofile = os.path.basename(ifile).split('.')[0] + '_mnf.tif'
                ofile = os.path.join(odir, ofile)

                self.showprocesslog('Exporting '+os.path.basename(ofile))
                export_gdal(ofile, odata, 'GTiff', piter=self.piter)

        elif 'Raster' in self.indata:
            dat = self.indata['Raster']
            odata, self.ev = mnf_calc(dat, ncmps, piter=self.piter,
                                      pprint=self.showprocesslog,
                                      noisetxt=noise)

        self.outdata['Raster'] = odata
        return True


# @profile
def get_noise(x2d, mask, noise=''):
    """
    Calculate noise dataset from original data.

    Parameters
    ----------
    x2d : numpy array
        Input array, of dimension (MxNxChannels).
    mask : numpy array
        mask of dimension (MxN).
    noise : str, optional
        Noise type to calculate. Can be 'diagonal', 'hv average' or ''.
        The default is ''.

    Returns
    -------
    nevals : numpy array
        Noise eigenvalues.
    nevecs : numpy array
        Noise eigenvectors.

    """
    mask = ~mask

    if noise == 'diagonal':
        t1 = x2d[:-1, :-1]
        t2 = x2d[1:, 1:]
        noise = ne.evaluate('t1-t2')

        mask2 = mask[:-1, :-1]*mask[1:, 1:]
        noise = noise[mask2]
        ncov = np.cov(noise.T)
    elif noise == 'hv average':
        t1 = x2d[:-1, :-1]
        t2 = x2d[1:, :-1]
        t3 = x2d[:-1, :-1]
        t4 = x2d[:-1, 1:]

        noise = ne.evaluate('(t1-t2+t3-t4)')
        mask2 = mask[:-1, :-1]*mask[1:, :-1]*mask[:-1, 1:]

        noise = noise[mask2]
        ncov = np.cov(noise.T)/4
    else:
        t1 = x2d[:-2, :-2]
        t2 = x2d[:-2, 1:-1]
        t3 = x2d[:-2, 2:]
        t4 = x2d[1:-1, :-2]
        t5 = x2d[1:-1, 1:-1]
        t6 = x2d[1:-1, 2:]
        t7 = x2d[2:, :-2]
        t8 = x2d[2:, 1:-1]
        t9 = x2d[2:, 2:]

        noise = ne.evaluate('(t1-2*t2+t3-2*t4+4*t5-2*t6+t7-2*t8+t9)')

        # noise = (x2d[:-2, :-2] - 2*x2d[:-2, 1:-1] + x2d[:-2, 2:]
        #          - 2*x2d[1:-1, :-2] + 4*x2d[1:-1, 1:-1] - 2*x2d[1:-1, 2:]
        #          + x2d[2:, :-2] - 2*x2d[2:, 1:-1] + x2d[2:, 2:])/9

        mask2 = (mask[:-2, :-2] * mask[:-2, 1:-1] * mask[:-2, 2:] *
                 mask[1:-1, :-2] * mask[1:-1, 1:-1] * mask[1:-1, 2:] *
                 mask[2:, :-2] * mask[2:, 1:-1] * mask[2:, 2:])

        noise = noise[mask2]
        ncov = np.cov(noise.T)/81

    # Calculate evecs and evals
    nevals, nevecs = np.linalg.eig(ncov)

    # return noise, mask2
    return nevals, nevecs


# @profile
def mnf_calc(dat, ncmps=None, noisetxt='hv average', pprint=print,
             piter=iter):
    """
    MNF Calculation.

    Parameters
    ----------
    dat : List
        List of PyGMI Data.
    ncmps : int or None, optional
        Number of components to use for filtering. The default is None
        (meaning all).
    noisetxt : txt, optional
        Noise type. Can be 'diagonal', 'hv average' or 'quad'. The default is
        'hv average'.
    pprint : function, optional
        Function for printing text. The default is print.
    piter : function, optional
        Iteration function, used for progressbars. The default is iter.

    Returns
    -------
    odata : list
        Output list of PyGMI Data.Can be forward or inverse transformed data.
    ev : numpy array
        Explained variance, from PCA.

    """
    x2d = []
    maskall = []
    for j in dat:
        x2d.append(j.data)
        maskall.append(j.data.mask)

    maskall = np.moveaxis(maskall, 0, -1)
    x2d = np.moveaxis(x2d, 0, -1)
    x2dshape = x2d.shape

    # if x2d.dtype != np.float64:
    #     x2d = x2d.astype(np.float32)

    # if x2d.dtype == np.uint16 or x2d.dtype == np.uint8:
    #     x2d = x2d.astype(np.int32)
    # elif x2d.dtype == np.uint32 or x2d.dtype == np.uint64:
    #     x2d = x2d.astype(np.int64)

    mask = maskall[:, :, 0]

    pprint('Calculating noise data...')
    nevals, nevecs = get_noise(x2d, mask, noisetxt)

    pprint('Calculating MNF...')
    Ln = np.power(nevals, -0.5)
    Ln = np.diag(Ln)

    # W = Ln @ nevecs.T
    W = np.dot(Ln, nevecs.T)

    x = x2d[~mask]
    del x2d

    Pnorm = np.dot(x, W.T)

    pca = IncrementalPCA(n_components=ncmps)

    iold = 0
    pprint('Fitting PCA')
    for i in piter(np.linspace(0, Pnorm.shape[0], 20, dtype=int)):
        if i == 0:
            continue
        pca.partial_fit(Pnorm[iold: i])
        iold = i

    pprint('Calculating PCA transform...')

    x2 = np.zeros((Pnorm.shape[0], pca.n_components_))
    iold = 0
    for i in piter(np.linspace(0, Pnorm.shape[0], 20, dtype=int)):
        if i == 0:
            continue
        x2[iold: i] = pca.transform(Pnorm[iold: i])
        iold = i

    del Pnorm
    ev = pca.explained_variance_

    if ncmps is not None:
        pprint('Calculating inverse MNF...')
        Winv = np.linalg.inv(W)
        P = pca.inverse_transform(x2)
        # x2 = (Winv @  P.T).T
        x2 = np.dot(P, Winv.T)
        del P

    datall = np.zeros(x2dshape, dtype=np.float32)
    datall[~mask] = x2
    datall = np.ma.array(datall, mask=maskall)

    del x2

    odata = copy.deepcopy(dat)
    for j, band in enumerate(odata):
        band.data = datall[:, :, j]

    del datall

    return odata, ev


# def mnf_calc2(x2d, maskall, ncmps=7, noisetxt='', pprint=print, piter=iter):
#     """MNF Filtering Copy."""
#     dim = x2d.shape[-1]
#     mask = maskall[:, :, 0]
#     x = x2d[~mask]

#     pprint('Calculating noise data...')
#     nevals, nevecs = get_noise(x2d, mask, noisetxt)

#     pprint('Calculating MNF...')
#     Ln = np.power(nevals, -0.5)
#     Ln = np.diag(Ln)

#     W = Ln @ nevecs.T
#     Winv = np.linalg.inv(W)

#     Pnorm = W @ x.T

#     pca = IncrementalPCA(n_components=ncmps)
#     P = pca.fit_transform(Pnorm.T)

#     pprint('Calculating inverse MNF...')
#     P = pca.inverse_transform(P)

#     x2 = (Winv @  P.T).T

#     rows, cols = mask.shape
#     datall = np.zeros([rows, cols, dim])
#     datall[~mask] = x2
#     datall = np.ma.array(datall, mask=maskall)

#     return datall


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt
    # import spectral as sp

    pbar = ProgressBarText()

    ifile = r'C:\Workdata\lithosphere\Cut-90-0824-.hdr'
    # ifile = r"C:\Workdata\Lithosphere\crash\033_0815-1111_ref_rect.hdr"
    ncmps = 10
    nodata = 0
    iraster = None

    dat = get_raster(ifile, nval=nodata, iraster=iraster, piter=pbar.iter)
    pmnf, ev = mnf_calc(dat, ncmps=ncmps, noisetxt='', piter=pbar.iter)

    # dat2 = []
    # maskall = []
    # for j in dat:
    #     dat2.append(j.data.astype(float))
    #     mask = j.data.mask
    #     maskall.append(mask)

    # maskall = np.moveaxis(maskall, 0, -1)
    # dat2 = np.moveaxis(dat2, 0, -1)

    # signal = sp.calc_stats(dat2)
    # noise = sp.noise_from_diffs(dat2)
    # mnfr = sp.mnf(signal, noise)
    # denoised = mnfr.denoise(dat2, num=ncmps)

    for i in [0, 5, 10, 13, 14, 15, 20, 25]:
        vmax = dat[i].data.max()
        vmin = dat[i].data.min()

        plt.figure(dpi=150)
        plt.title('█████████████████Old dat2 band'+str(i))
        plt.imshow(dat[i].data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

        plt.figure(dpi=150)
        plt.title('New MNF denoised band'+str(i))
        # plt.imshow(pmnf[i].data, vmin=vmin, vmax=vmax)
        plt.imshow(pmnf[i].data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

        # plt.figure(dpi=150)
        # plt.title('SPy MNF denoised band'+str(i))
        # plt.imshow(np.ma.array(denoised[:, :, i], mask=mask), vmin=vmin,
        #             vmax=vmax)
        # plt.colorbar()
        # plt.show()

    return


def _testfn2():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import spectral as sp

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
        dat2.append(j.data.astype(float))
        mask = j.data.mask
        maskall.append(mask)

    maskall = np.moveaxis(maskall, 0, -1)
    dat2 = np.moveaxis(dat2, 0, -1)

    # signal = sp.calc_stats(dat2)
    # noise = sp.noise_from_diffs(dat2)
    # mnfr = sp.mnf(signal, noise)
    # denoised = mnfr.denoise(dat2, num=ncmps)
    # scov = noise.cov

    for i in ['diagonal', 'hv average', '']:
        noise, maskp = get_noise(dat2, mask=mask, noise=i)

        n2 = np.zeros(maskp.shape+(dat2.shape[-1],))
        n2[maskp] = noise
        n2 = np.ma.array(n2[:, :, 0], mask=~maskp)

        vmin = n2.mean()-2*n2.std()
        vmax = n2.mean()+2*n2.std()
        plt.title(i)
        plt.grid(True)
        plt.imshow(n2[600:800, 600:800], vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()
        plt.hist(n2.flatten(), 50)
        plt.show()
        plt.plot(n2[600])
        plt.show()

    # pmnf = mnf_calc(dat2, maskall, ncmps=ncmps, noisetxt='diagonal')
    # pmnf, ev = mnf_calc(dat, ncmps=None, noisetxt='diagonal')

    return


if __name__ == "__main__":
    _testfn()
