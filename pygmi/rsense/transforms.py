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
import os
import copy
from math import ceil
import gc

import numpy as np
from PyQt5 import QtWidgets, QtCore
from sklearn.decomposition import PCA
import numexpr as ne
import matplotlib.pyplot as plt

from pygmi.raster.iodefs import get_raster
from pygmi.misc import ProgressBarText
from pygmi.raster.iodefs import export_gdal
import pygmi.menu_default as menu_default
from pygmi.misc import PTime, getmem


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


def get_noise(x2d, mask, noise=''):
    """
    Calculates noise dataset from original data.

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
        Noise eigen values.
    nevecs : numpy array
        Noise eigen vectors.

    """
    mask = ~mask

    getmem('noise1')

    ttt = PTime()

    if noise == 'diagonal':
        t1 = x2d[:-1, :-1]
        t2 = x2d[1:, 1:]
        noise = ne.evaluate('t1-t2')
        # noise = x2d[:-1, :-1] - x2d[1:, 1:]

        mask2 = mask[:-1, :-1]*mask[1:, 1:]
        noise = noise[mask2]
        ncov = np.cov(noise.T)/2
    elif noise == 'hv average':
        t1 = x2d[:-1, :-1]
        t2 = x2d[1:, :-1]
        t3 = x2d[:-1, :-1]
        t4 = x2d[:-1, 1:]

        noise = ne.evaluate('(t1-t2+t3-t4)/2')

        # vdiff = x2d[:-1] - x2d[1:]
        # hdiff = x2d[:, :-1] - x2d[:, 1:]
        # noise = (vdiff[:, :-1]+hdiff[:-1])/2

        getmem('noise1c')

        # mask2a = mask[:-1]*mask[1:]
        # mask2b = mask[:, :-1]*mask[:, 1:]
        # mask2 = mask2a[:, :-1]*mask2b[:-1]

        mask2 = mask[:-1, :-1]*mask[1:, :-1]*mask[:-1, 1:]
        noise = noise[mask2]
        ncov = np.cov(noise.T)/2

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

        noise = ne.evaluate('(t1-2*t2+t3-2*t4+4*t5-2*t6+t7-2*t8+t9)/9')

        # noise = (x2d[:-2, :-2] - 2*x2d[:-2, 1:-1] + x2d[:-2, 2:]
        #          - 2*x2d[1:-1, :-2] + 4*x2d[1:-1, 1:-1] - 2*x2d[1:-1, 2:]
        #          + x2d[2:, :-2] - 2*x2d[2:, 1:-1] + x2d[2:, 2:])/9

        mask2 = (mask[:-2, :-2] * mask[:-2, 1:-1] * mask[:-2, 2:] *
                 mask[1:-1, :-2] * mask[1:-1, 1:-1] * mask[1:-1, 2:] *
                 mask[2:, :-2] * mask[2:, 1:-1] * mask[2:, 2:])

        noise = noise[mask2]
        ncov = np.cov(noise.T)/2

    # Calculate evecs and evals
    nevals, nevecs = np.linalg.eig(ncov)

    getmem('noise3')

    return nevals, nevecs


def mnf_calc(dat, ncmps=None, noisetxt='hv average', pprint=print,
             piter=iter):
    """MNF Filtering"""

    getmem('mnf in')

    x2d = []
    maskall = []
    for j in dat:
        x2d.append(j.data)
        maskall.append(j.data.mask)
    maskall = np.moveaxis(maskall, 0, -1)
    x2d = np.moveaxis(x2d, 0, -1)

    getmem('0')

    if x2d.dtype == np.uint16 or x2d.dtype == np.uint8:
        x2d = x2d.astype(np.int32)
    elif x2d.dtype == np.uint32 or x2d.dtype == np.uint64:
        x2d = x2d.astype(np.int64)

    mask = maskall[:, :, 0]

    getmem('1')

    pprint('Calculating noise data...')
    nevals, nevecs = get_noise(x2d, mask, noisetxt)

    getmem('2')

    pprint('Calculating MNF...')
    Ln = np.power(nevals, -0.5)
    Ln = np.diag(Ln)

    getmem('3')

    W = Ln @ nevecs.T

    x = x2d[~mask]

    getmem('4')

    # breakpoint()
    Pnorm = np.dot(W, x.T)
    # Pnorm = W @ x.T

    getmem('5')
    pca = PCA(n_components=ncmps)
    x2 = pca.fit_transform(Pnorm.T)
    ev = pca.explained_variance_

    getmem('6')

    if ncmps is not None:
        pprint('Calculating inverse MNF...')
        Winv = np.linalg.inv(W)
        P = pca.inverse_transform(x2)
        x2 = (Winv @  P.T).T

    getmem('7')

    datall = np.zeros(x2d.shape, dtype=np.float32)
    datall[~mask] = x2
    datall = np.ma.array(datall, mask=maskall)

    del x2
    getmem('8')

    odata = copy.deepcopy(dat)
    for j, band in enumerate(odata):
        band.data = datall[:, :, j]

    del datall
    getmem('9')

    return odata, ev


def mnf_calc2(x2d, maskall, ncmps=7, noisetxt='', pprint=print, piter=iter):
    """MNF Filtering"""

    dim = x2d.shape[-1]
    mask = maskall[:, :, 0]
    x = x2d[~mask]

    pprint('Calculating noise data...')
    nevals, nevecs = get_noise(x2d, mask, noisetxt)

    pprint('Calculating MNF...')
    Ln = np.power(nevals, -0.5)
    Ln = np.diag(Ln)

    W = Ln @ nevecs.T
    Winv = np.linalg.inv(W)

    Pnorm = W @ x.T

    pca = PCA(n_components=ncmps)
    P = pca.fit_transform(Pnorm.T)

    pprint('Calculating inverse MNF...')
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

    ttt = PTime()

    signal = sp.calc_stats(dat2)
    noise = sp.noise_from_diffs(dat2)
    mnfr = sp.mnf(signal, noise)
    denoised = mnfr.denoise(dat2, num=ncmps)
    scov = noise.cov

    mask = ~mask
    x2d = dat2
    noise = x2d[:-1, :-1] - x2d[1:, 1:]
    # mask2 = mask[:-1, :-1]*mask[1:, 1:]
    mask2 = mask[:-1, :-1]
    noise = noise[mask2]
    ncov = np.cov(noise.T)/2

    print('Calculating MNF')
    pmnf, ev = mnf_calc2(dat, ncmps=ncmps, noisetxt='diagonal')
    # pmnf = mnf_calc(dat2, maskall, ncmps=ncmps, noisetxt='diagonal')
    # pmnf, ev = mnf_calc(dat, ncmps=None, noisetxt='diagonal')

    ttt.since_last_call()

    for i in [0, 5, 10, 13, 14, 15, 20, 25]:
        vmax = dat[i].data.max()
        vmin = dat[i].data.min()

        plt.title('█████████████████Old dat2 band'+str(i))
        plt.imshow(dat[i].data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

        plt.title('SPy MNF denoised band'+str(i))
        plt.imshow(np.ma.array(denoised[:, :, i], mask=~mask), vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

        plt.title('New MNF denoised band'+str(i))
        # plt.imshow(pmnf[i].data, vmin=vmin, vmax=vmax)
        plt.imshow(pmnf[:, :, i], vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

    return


def _testfn2():
    """Test routine."""
    import sys

    from matplotlib import rcParams

    rcParams['figure.dpi'] = 300

    pbar = ProgressBarText()

    idir = r'C:\Workdata\Lithosphere\batch'
    ifile = r'C:\Workdata\lithosphere\Cut-90-0824-.hdr'
    ifile = r'C:\Workdata\lithosphere\cut-087-0824.hdr'

    data = get_raster(ifile, nval=0, iraster=None, piter=pbar.iter)

    getmem('testfn')

    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes
    tmp = MNF()
    tmp.indata['Raster'] = data
    tmp.settings()


if __name__ == "__main__":
    _testfn2()


"""
Memory check: testfn, RAM memory used: 8.7 GB (27.3%)
Memory check: mnf in, RAM memory used: 8.7 GB (27.3%)
Memory check: 0, RAM memory used: 13.2 GB (41.5%)
Memory check: 1, RAM memory used: 16.3 GB (51.0%)
Calculating noise data...
Memory check: noise1, RAM memory used: 16.3 GB (51.0%)
Memory check: noise3, RAM memory used: 16.4 GB (51.2%)
Memory check: 2, RAM memory used: 6.5 GB (20.3%)
Calculating MNF...
Memory check: 3, RAM memory used: 6.5 GB (20.3%)
Memory check: 4, RAM memory used: 19.3 GB (60.4%)
Memory check: 5, RAM memory used: 22.1 GB (69.3%)
Memory check: 6, RAM memory used: 11.4 GB (35.6%)
Memory check: 7, RAM memory used: 11.4 GB (35.6%)
Memory check: 8, RAM memory used: 7.3 GB (22.9%)
Memory check: 9, RAM memory used: 13.8 GB (43.2%)

"""
