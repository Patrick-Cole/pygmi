# -----------------------------------------------------------------------------
# Name:        change.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2023 Council for Geoscience
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
"""Calculate change detection indices."""

import math
import os
import sys
import numpy as np
from numba import jit
from PyQt5 import QtWidgets, QtCore

from pygmi import menu_default
from pygmi.rsense.iodefs import get_from_rastermeta
from pygmi.raster.misc import lstack
from pygmi.misc import BasicModule


class CalculateChange(BasicModule):
    """Calculate Change Indices."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lw_indices = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        btn_invert = QtWidgets.QPushButton('Invert Selection')
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.rsense.change')
        lbl_ratios = QtWidgets.QLabel('Indices:')

        self.lw_indices.setSelectionMode(self.lw_indices.MultiSelection)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Calculate Change Indices')

        gl_main.addWidget(lbl_ratios, 1, 0, 1, 1)
        gl_main.addWidget(self.lw_indices, 1, 1, 1, 1)
        gl_main.addWidget(btn_invert, 2, 0, 1, 2)

        gl_main.addWidget(helpdocs, 6, 0, 1, 1)
        gl_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.lw_indices.clicked.connect(self.set_selected_indices)
        btn_invert.clicked.connect(self.invert_selection)

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
        tmp = []
        if 'RasterFileList' not in self.indata:
            self.showlog('No batch file list detected.')
            return False

        self.setindices()

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.lw_indices)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        flist = self.indata['RasterFileList']

        ilist = []
        for i in self.lw_indices.selectedItems():
            ilist.append(i.text()[2:])

        if not ilist:
            self.showlog('You need to select an index to calculate.')
            return False

        datfin = calc_change(flist, ilist, showlog=self.showlog,
                             piter=self.piter)

        if not datfin:
            return False

        self.outdata['Raster'] = datfin

        return True

    def setindices(self):
        """
        Set the available indices.

        Returns
        -------
        None.

        """
        ilist = ['Difference', 'Mean', 'Standard Deviation',
                 'Coefficient of Variation',
                 'Spectral Angle Mapper']

        self.lw_indices.clear()
        self.lw_indices.addItems(ilist)

        for i in range(self.lw_indices.count()):
            item = self.lw_indices.item(i)
            item.setSelected(True)
            item.setText('\u2713 ' + item.text())

    def invert_selection(self):
        """
        Invert the selected indices.

        Returns
        -------
        None.

        """
        for i in range(self.lw_indices.count()):
            item = self.lw_indices.item(i)
            item.setSelected(not item.isSelected())

        self.set_selected_indices()

    def set_selected_indices(self):
        """
        Set the selected indices.

        Returns
        -------
        None.

        """
        for i in range(self.lw_indices.count()):
            item = self.lw_indices.item(i)
            if item.isSelected():
                item.setText('\u2713' + item.text()[1:])
            else:
                item.setText(' ' + item.text()[1:])


def calc_change(flist, ilist=None, showlog=print, piter=iter):
    """
    Calculate Change Indices.

    Parameters
    ----------
    flist : list of RasterMeta.
        List of batch file list data.
    ilist : list, optional
        List of strings describing index to calculate.
    showlog : function, optional
        Display information. The default is print.
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    datfin : list of PyGMI Data
        List of PyGMI Data.

    """
    if len(flist) < 2:
        showlog('You need a minimum of two datasets.')
        return None

    meandat = None
    std = None
    datfin = []

    if 'Mean' in ilist:
        meandat, cnt, M = calc_mean(flist, showlog, piter)

    if 'Standard Deviation' in ilist:
        showlog('Calculating STD...')

        if meandat is None:
            meandat, cnt, M = calc_mean(flist, showlog, piter)

        std = {}
        for i in meandat:
            std[i] = meandat[i].copy(True)
            std[i].data = stddev(M[i], cnt[i])
            std[i].dataid += '_STD'
        datfin += list(std.values())

    if 'Coefficient of Variation' in ilist:
        showlog('Calculating CV...')

        if meandat is None:
            meandat, cnt, M = calc_mean(flist, showlog, piter)
        if std is None:
            std = {}
            for i in meandat:
                std[i] = meandat[i].copy(True)
                std[i].data = stddev(M[i], cnt[i])

        cv = {}
        for i in meandat:
            cv[i] = meandat[i].copy(True)
            cv[i].data = coefv(meandat[i].data, std[i].data)
            cv[i].dataid += '_CV'

        datfin += list(cv.values())

    if 'Mean' in ilist:
        for i in meandat:
            meandat[i].dataid += '_MEAN'
        datfin += list(meandat.values())

    if 'Spectral Angle Mapper' in ilist and len(flist) != 2:
        showlog('Only two datasets allowed for SAM.')
        # Add loop for maximum angle deviation and std dev.
    elif 'Spectral Angle Mapper' in ilist:
        sam1 = calc_sam(flist, showlog, piter)
        sam1.dataid += '_SAM'
        datfin += [sam1]

    if 'Difference' in ilist and len(flist) != 2:
        showlog('Only two datasets allowed for difference.')
    elif 'Difference' in ilist:
        showlog('Calculating difference...')

        dat1, dat2 = match_data(flist, showlog=showlog, piter=piter)

        diff = [i.copy() for i in dat1]

        for i, dband in enumerate(diff):
            dband.data = dat2[i].data-dat1[i].data
            dband.dataid += '_DIFF'
        datfin += diff

    return datfin


def calc_mean(flist, showlog=print, piter=iter):
    """
    Load data and calculate iterative Mean.

    Parameters
    ----------
    flist : list of RasterMeta
        List of batch file list data.
    showlog : function, optional
        Display information. The default is print.
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    meandat : dictionary of PyGMI Data.
        PyGMI Data representing means.
    cnt : dictionary of numpy arrays
        Count of values which made up mean.
    M : dictionary of numpy arrays
        Variance parameter, where Variance = M/cnt.

    """
    showlog('Calculating mean...')
    tmp = get_from_rastermeta(flist[0], piter=piter, showlog=showlog)

    meandat = {}
    for val in tmp:
        meandat[val.dataid] = val

    # Init variables using first file above.
    cnt = {}
    M = {}

    for i in meandat:
        cnt[i] = None
        M[i] = None
        cnt[i] = meandat[i].copy()
        M[i] = meandat[i].copy()
        cnt[i].data = np.ones_like(cnt[i].data)
        M[i].data = np.zeros_like(M[i].data)

    # Iteratively calculate stats
    for ifile in piter(flist[1:]):
        tmp = get_from_rastermeta(ifile, piter=piter, showlog=showlog)
        dat = {}
        for val in tmp:
            dat[val.dataid] = val

        for i in meandat:
            if i not in dat:
                showlog(f'{i} not in first dataset, skipping.')
                continue

            ltmp = [meandat[i], dat[i], cnt[i], M[i]]
            ltmp = lstack(ltmp, showlog=showlog, piter=piter,
                          checkdataid=False)
            meandat[i], dat[i], cnt[i], M[i] = ltmp

            tmp = imean(meandat[i].data, dat[i].data, cnt[i].data, M[i].data)
            meandat[i].data, cnt[i].data, M[i].data = tmp

    for i in cnt:
        cnt[i] = cnt[i].data
        M[i] = M[i].data

    return meandat, cnt, M


def calc_sam(flist, showlog=print, piter=iter):
    """
    Load data and calculate spectral angle between two times.

    Parameters
    ----------
    flist : list of RasterMeta.
        List of batch file list data.
    showlog : function, optional
        Display information. The default is print.
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    angle : PyGMI Data
        PyGMI Data of SAM angles.

    """
    showlog('Calculating SAM...')

    dat1, dat2 = match_data(flist, showlog=showlog, piter=piter)

    dat1b = []
    for j in dat1:
        dat1b.append(j.data)

    dat2b = []
    for j in dat2:
        dat2b.append(j.data)

    dat1b = np.array(dat1b)
    dat1b = np.moveaxis(dat1b, 0, -1)
    dat2b = np.array(dat2b)
    dat2b = np.moveaxis(dat2b, 0, -1)

    # Init variables
    angle = dat1[0].copy(True)
    angle.data = angle.data.astype(float)
    angle.data *= 0.

    rows, cols = angle.data.shape

    for i in piter(range(rows)):
        for j in range(cols):
            s1 = dat1b[i, j]
            s2 = dat2b[i, j]
            angle.data[i, j] = sam(s1, s2)

    angle.nodata = 0.
    angle.data.mask = dat1[0].data.mask
    angle.data = angle.data.filled(0.)
    angle.data = np.ma.array(angle.data, mask=dat1[0].data.mask)

    return angle


def coefv(mean, std):
    """
    Calculate coefficient of variation.

    Parameters
    ----------
    mean : numpy array
        numpy array of mean values.
    std : numpy array
        numpy array of standard deviation values.

    Returns
    -------
    cv : numpy array
        Array of coefficient of variation values.

    """
    # Sqrt to convert variance to standard deviation
    cv = std/mean

    tmp = cv.compressed()
    perc1 = np.percentile(tmp, 1)
    cv[cv < perc1] = perc1

    perc99 = np.percentile(tmp, 99)
    cv[cv > perc99] = perc99

    return cv


def imean(mean, newdat, cnt=None, M=None):
    """
    Calculate mean and variance parameters.

    Parameters
    ----------
    mean : numpy array
        existing mean values.
    newdat : numpy array
        new data to be added to mean..
    cnt : numpy array, optional
        cnt of values which made up mean. The default is None.
    M : numpy array, optional
        Variance parameter, where Variance = M/cnt. The default is None.

    Returns
    -------
    mean : numpy array
        Updated mean of data.
    cnt : numpy array
        Updated cnt of values which made up mean.
    M : numpy array
        Updated variance parameter, where Variance = M/cnt.

    """
    if cnt is None:
        cnt = np.ones_like(mean)
    if M is None:
        M = np.zeros_like(mean)
    mean = mean.astype(float)
    newdat = newdat.astype(float)

    n1 = cnt
    cnt = cnt + 1
    n = cnt
    delta = newdat - mean
    delta_n = delta / n
    term1 = delta * delta_n * n1
    mean = mean + delta_n
    M = M + term1

    return mean, cnt, M


def match_data(flist, showlog=print, piter=iter):
    """
    Match two datasets.

    Parameters
    ----------
    flist : list of RasterMeta
        List of batch file list data.
    showlog : function, optional
        Display information. The default is print.
    piter : function, optional
        Progress bar iterator. The default is iter.

    Returns
    -------
    dat1 : list of PyGMI Data
        First dataset with matched bands only.
    dat2 : list of PyGMI Data
        Second dataset with matched bands only.

    """
    if len(flist) > 2:
        showlog('You have more than two datasets being matched. '
                'Only the first two will be used.')

    tnames = list(set(flist[0].tnames).intersection(set(flist[1].tnames)))

    dat1 = get_from_rastermeta(flist[0], piter=piter, showlog=showlog,
                               tnames=tnames)
    dat2 = get_from_rastermeta(flist[1], piter=piter, showlog=showlog,
                               tnames=tnames)

    tmp = lstack(dat1+dat2, showlog=showlog, piter=piter, checkdataid=False)

    dat1 = tmp[:len(tnames)]
    dat2 = tmp[len(tnames):]

    if dat1[0].datetime > dat2[0].datetime:
        dat1, dat2 = dat2, dat1

    return dat1, dat2


@jit(nopython=True)
def sam(s1, s2):
    """
    Calculate Spectral Angle Mapper (SAM).

    Parameters
    ----------
    s1 : numpy array
        Spectrum 1.
    s2 : numpy array
        Spectrum 2.

    Returns
    -------
    result : numpy array
        Output angles.

    """
    s1a = s1.astype('d')
    s2a = s2.astype('d')

    num = np.dot(s1a, s2a)
    denom = np.sqrt(np.sum(s1a**2))*np.sqrt(np.sum(s2a**2))

    if denom == 0.:
        result = 0.0
    else:
        result = math.acos(num/denom)

    return result


def scm(s1, s2):
    """
    SCM or MSAM.

    Parameters
    ----------
    s1 : numpy array
        Spectrum 1.
    s2 : numpy array
        Spectrum 2.

    Returns
    -------
    result : numpy array
        Output angles.

    """
    s1 = s1.astype('d')
    s2 = s2.astype('d')

    s1a = s1-s1.mean()
    s2a = s2-s2.mean()

    num = np.dot(s1a, s2a)
    denom = np.sqrt(np.sum(s1a**2))*np.sqrt(np.sum(s2a**2))

    if denom == 0.:
        result = -1.0
    else:
        result = num/denom

    return result


def stddev(M, cnt):
    """
    Calculate std deviation.

    Parameters
    ----------
    M : numpy array
        Variance parameter, where Variance = M/cnt.
    cnt : numpy array
        cnt of values which made up mean.

    Returns
    -------
    std : numpy array
        Calculated standard deviation.

    """
    var = M/cnt
    std = np.sqrt(var)

    return std


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt
    from pygmi.rsense.iodefs import ImportBatch

    idir = r'E:\WorkProjects\ST-2020-1339 Landslides\change\ratios'
    idir = r'E:\WorkProjects\ST-2020-1339 Landslides\change\mosaic\ratios'
    os.chdir(r'E:\\')

    app = QtWidgets.QApplication(sys.argv)

    tmp1 = ImportBatch()
    tmp1.idir = idir
    tmp1.get_sfile(True)
    tmp1.settings()

    tmp2 = CalculateChange()
    tmp2.indata = tmp1.outdata
    tmp2.settings()

    dat2 = tmp2.outdata['Raster']
    for i in dat2:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        vmin = i.data.mean()-2*i.data.std()
        vmax = i.data.mean()+2*i.data.std()
        plt.imshow(i.data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    _testfn()
