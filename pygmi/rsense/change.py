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
from collections.abc import Callable, Iterable

import numpy as np
from numba import jit
from numpy.typing import NDArray
from PySide6 import QtWidgets

from pygmi.misc import BasicModule
from pygmi.raster.datatypes import Data, RasterMeta
from pygmi.raster.misc import lstack
from pygmi.rsense.iodefs import get_from_rastermeta


class CalculateChange(BasicModule):
    """
    GUI to calculate change indices.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lw_indices = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """Set up UI."""
        self.buttonbox.htmlfile = "rsense.dm.change.html#calculate-change-indices"
        gl_main = QtWidgets.QGridLayout(self)
        btn_invert = QtWidgets.QPushButton("Invert Selection")
        lbl_ratios = QtWidgets.QLabel("Indices:")

        self.lw_indices.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )

        ilist = [
            "Difference",
            "Mean",
            "Standard Deviation",
            "Coefficient of Variation",
            "Spectral Angle Mapper",
        ]

        self.lw_indices.addItems(ilist)

        self.setWindowTitle("Calculate Change Indices")

        gl_main.addWidget(lbl_ratios, 1, 0, 1, 1)
        gl_main.addWidget(self.lw_indices, 1, 1, 1, 1)
        gl_main.addWidget(btn_invert, 2, 0, 1, 2)

        gl_main.addWidget(self.buttonbox, 6, 0, 1, 2)

        # self.lw_indices.clicked.connect(self.set_selected_indices)
        btn_invert.clicked.connect(self.invert_selection)

    def settings(self, nodialog: bool = False) -> bool:
        """
        Entry point into item.

        Parameters
        ----------
        nodialog
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if "RasterFileList" not in self.indata:
            self.showlog("No batch file list detected.")
            return False

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def saveproj(self):
        """Save project data from class."""
        self.saveobj(self.lw_indices)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        """
        flist = self.indata["RasterFileList"]

        ilist = []
        for i in self.lw_indices.selectedItems():
            ilist.append(i.text())

        if not ilist:
            self.showlog("You need to select an index to calculate.")
            return False

        datfin = calc_change(flist, ilist, showlog=self.showlog, piter=self.piter)

        if not datfin:
            return False

        self.outdata["Raster"] = datfin

        return True

    def invert_selection(self):
        """Invert the selected indices."""
        for i in range(self.lw_indices.count()):
            item = self.lw_indices.item(i)
            item.setSelected(not item.isSelected())


def calc_change(
    flist: list[RasterMeta],
    ilist: list[str] | None = None,
    showlog: Callable[..., None] = print,
    piter: Iterable = iter,
) -> list[Data]:
    """
    Calculate Change Indices.

    Parameters
    ----------
    flist
        List of batch file list data.
    ilist
        List of strings describing index to calculate.
    showlog
        Display information. The default is print.
    piter
        Progress bar iterator. The default is iter.

    Returns
    -------
    list of Data
        List of PyGMI Data.

    """
    if len(flist) < 2:
        showlog("You need a minimum of two datasets.")
        return None

    meandat = {}
    std = None
    datfin = []
    M = []
    cnt = []

    if (
        "Standard Deviation" in ilist
        or "Coefficient of Variation" in ilist
        or "Mean" in ilist
    ):
        meandat, cnt, M = calc_mean(flist, showlog, piter)

    if "Standard Deviation" in ilist:
        showlog("Calculating STD...")

        std = {}
        for i in meandat:
            std[i] = meandat[i].copy(resetmeta=True)
            std[i].data = stddev(M[i], cnt[i])
            std[i].dataid += "_STD"
        datfin += list(std.values())

    if "Coefficient of Variation" in ilist:
        showlog("Calculating CV...")

        if std is None:
            std = {}
            for i in meandat:
                std[i] = meandat[i].copy(resetmeta=True)
                std[i].data = stddev(M[i], cnt[i])

        cv = {}
        for i in meandat:
            cv[i] = meandat[i].copy(resetmeta=True)
            cv[i].data = coefv(meandat[i].data, std[i].data)
            cv[i].dataid += "_CV"

        datfin += list(cv.values())

    if "Mean" in ilist:
        for i in meandat:
            meandat[i].dataid += "_MEAN"
        datfin += list(meandat.values())

    if "Spectral Angle Mapper" in ilist and len(flist) != 2:
        showlog("Only two datasets allowed for SAM.")
        # Add loop for maximum angle deviation and std dev.
    elif "Spectral Angle Mapper" in ilist:
        sam1 = calc_sam(flist, showlog, piter)
        sam1.dataid += "_SAM"
        datfin += [sam1]

    if "Difference" in ilist and len(flist) != 2:
        showlog("Only two datasets allowed for difference.")
    elif "Difference" in ilist:
        showlog("Calculating difference...")

        dat1, dat2 = match_data(flist, showlog=showlog, piter=piter)
        if dat1 is not None:
            diff = [i.copy() for i in dat1]

            for i, dband in enumerate(diff):
                dband.data = dat2[i].data - dat1[i].data
                dband.dataid += "_DIFF"
            datfin += diff

    return datfin


def calc_mean(
    flist: list[RasterMeta],
    showlog: Callable[..., None] = print,
    piter: Iterable = iter,
) -> tuple[dict[str, Data], dict[str, NDArray], dict[str, NDArray]]:
    """
    Load data and calculate iterative Mean.

    Parameters
    ----------
    flist
        List of batch file data.
    showlog
        Display information. The default is print.
    piter
        Progress bar iterator. The default is iter.

    Returns
    -------
    meandat : dictionary of pygmi.raster.datatypes.Data.
        PyGMI Data representing means.
    cnt : dictionary of ndarrays
        Count of values which made up mean.
    M : dictionary of ndarrays
        Variance parameter, where Variance = M/cnt.

    """
    showlog("Calculating mean...")
    tmp = get_from_rastermeta(flist[0], piter=piter, showlog=showlog)

    meandat = {}
    for val in tmp:
        meandat[val.dataid] = val

    # Init variables using first file above.
    cnt = {}
    M = {}

    for i, value in meandat.items():
        cnt[i] = value.copy()
        M[i] = value.copy()
        cnt[i].data = np.ones_like(cnt[i].data)
        M[i].data = np.zeros_like(M[i].data)

    # Iteratively calculate stats
    for ifile in piter(flist[1:]):
        tmp = get_from_rastermeta(ifile, piter=piter, showlog=showlog)
        dat = {}
        for val in tmp:
            dat[val.dataid] = val

        for i, meandati in meandat.items():
            if i not in dat:
                showlog(f"{i} not in new dataset, skipping.")
                continue

            ltmp = [meandati, dat[i], cnt[i], M[i]]
            ltmp = lstack(ltmp, showlog=showlog, piter=piter, checkdataid=False)
            meandat[i], dat[i], cnt[i], M[i] = ltmp

            tmp = imean(meandati.data, dat[i].data, cnt[i].data, M[i].data)
            meandat[i].data, cnt[i].data, M[i].data = tmp

    for i, cnti in cnt.items():
        cnt[i] = cnti.data
        M[i] = M[i].data

    return meandat, cnt, M


def calc_sam(
    flist: list[RasterMeta],
    showlog: Callable[..., None] = print,
    piter: Iterable = iter,
) -> Data:
    """
    Load data and calculate spectral angle between two times.

    Parameters
    ----------
    flist
        List of batch file list data.
    showlog
        Display information. The default is print.
    piter
        Progress bar iterator. The default is iter.

    Returns
    -------
    pygmi.raster.datatypes.Data
        PyGMI Data of SAM angles.

    """
    showlog("Calculating SAM...")

    dat1, dat2 = match_data(flist, showlog=showlog, piter=piter)
    if dat1 is None:
        return []

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
    angle = dat1[0].copy(resetmeta=True)
    angle.data = angle.data.astype(float)
    angle.data *= 0.0

    rows, cols = angle.data.shape

    for i in piter(range(rows)):
        for j in range(cols):
            s1 = dat1b[i, j]
            s2 = dat2b[i, j]
            angle.data[i, j] = sam(s1, s2)

    angle.nodata = 0.0
    angle.data.mask = dat1[0].data.mask
    angle.data = angle.data.filled(0.0)
    angle.data = np.ma.array(angle.data, mask=dat1[0].data.mask)

    return angle


def coefv(mean: NDArray, std: NDArray) -> NDArray:
    """
    Calculate coefficient of variation.

    Parameters
    ----------
    mean
        numpy array of mean values.
    std
        numpy array of standard deviation values.

    Returns
    -------
    ndarray
        Array of coefficient of variation values.

    """
    # Sqrt to convert variance to standard deviation
    cv = std / mean

    tmp = cv.compressed()
    perc1 = np.percentile(tmp, 1)
    cv[cv < perc1] = perc1

    perc99 = np.percentile(tmp, 99)
    cv[cv > perc99] = perc99

    return cv


def imean(
    mean: NDArray, newdat: NDArray, cnt: NDArray | None = None, M: NDArray | None = None
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculate mean and variance parameters.

    Parameters
    ----------
    mean
        existing mean values.
    newdat
        new data to be added to mean..
    cnt
        cnt of values which made up mean. The default is None.
    M
        Variance parameter, where Variance = M/cnt. The default is None.

    Returns
    -------
    mean : ndarray
        Updated mean of data.
    cnt : ndarray
        Updated cnt of values which made up mean.
    M : ndarray
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


def match_data(
    flist: list[Data | RasterMeta],
    showlog: Callable[..., None] = print,
    piter: Iterable = iter,
) -> tuple[list[Data], list[Data]]:
    """
    Match two datasets.

    This routine also puts the datasets in order of date.

    Parameters
    ----------
    flist
        List of batch file list data.
    showlog
        Display information. The default is print.
    piter
        Progress bar iterator. The default is iter.

    Returns
    -------
    dat1 : list of Data
        First dataset with matched bands only.
    dat2 : list of Data
        Second dataset with matched bands only.

    """
    if len(flist) > 2:
        showlog(
            "You have more than two datasets being matched. "
            "Only the first two will be used."
        )

    if isinstance(flist[0], list):
        tnames1 = [i.dataid for i in flist[0]]
        tnames2 = [i.dataid for i in flist[1]]
        tnames = list(set(tnames1).intersection(set(tnames2)))
    else:
        tnames = list(set(flist[0].tnames).intersection(set(flist[1].tnames)))

    if not tnames:
        showlog("Error: Could not find common band names.")
        return None, None

    dat1 = get_from_rastermeta(flist[0], piter=piter, showlog=showlog, tnames=tnames)
    dat2 = get_from_rastermeta(flist[1], piter=piter, showlog=showlog, tnames=tnames)

    tmp = lstack(dat1 + dat2, showlog=showlog, piter=piter, checkdataid=False)

    dat1 = tmp[: len(tnames)]
    dat2 = tmp[len(tnames) :]

    if dat1[0].datetime > dat2[0].datetime:
        dat1, dat2 = dat2, dat1

    return dat1, dat2


@jit(nopython=True)
def sam(s1: NDArray, s2: NDArray) -> NDArray:
    """
    Calculate Spectral Angle Mapper (SAM).

    Parameters
    ----------
    s1
        Spectrum 1.
    s2
        Spectrum 2.

    Returns
    -------
    ndarray
        Output angles.

    """
    s1a = s1.astype("d")
    s2a = s2.astype("d")

    num = np.dot(s1a, s2a)
    denom = np.sqrt(np.sum(s1a**2)) * np.sqrt(np.sum(s2a**2))

    if denom == 0.0:
        result = 0.0
    else:
        result = math.acos(num / denom)

    return result


def scm(s1: NDArray, s2: NDArray) -> NDArray:
    """
    SCM or MSAM.

    Parameters
    ----------
    s1
        Spectrum 1.
    s2
        Spectrum 2.

    Returns
    -------
    ndarray
        Output angles.

    """
    s1 = s1.astype("d")
    s2 = s2.astype("d")

    s1a = s1 - s1.mean()
    s2a = s2 - s2.mean()

    num = np.dot(s1a, s2a)
    denom = np.sqrt(np.sum(s1a**2)) * np.sqrt(np.sum(s2a**2))

    if denom == 0.0:
        result = -1.0
    else:
        result = num / denom

    return result


def stddev(M: NDArray, cnt: NDArray) -> NDArray:
    """
    Calculate std deviation.

    Parameters
    ----------
    M
        Variance parameter, where Variance = M/cnt.
    cnt
        cnt of values which made up mean.

    Returns
    -------
    ndarray
        Calculated standard deviation.

    """
    var = M / cnt
    std = np.sqrt(var)

    return std


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt

    from pygmi.rsense.iodefs import ImportBatch

    idir = r"D:\workdata\PyGMI Test Data\Remote Sensing\change\ratios"
    os.chdir(idir)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    tmp1 = ImportBatch()
    tmp1.idir = idir
    tmp1.get_sfile(True)
    tmp1.settings()

    tmp2 = CalculateChange()
    tmp2.indata = tmp1.outdata
    tmp2.settings()

    dat2 = tmp2.outdata["Raster"]
    for i in dat2:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        vmin = i.data.mean() - 2 * i.data.std()
        vmax = i.data.mean() + 2 * i.data.std()
        plt.imshow(i.data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    _testfn()
