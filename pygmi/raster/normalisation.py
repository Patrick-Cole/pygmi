# -----------------------------------------------------------------------------
# Name:        normalisation.py (part of PyGMI)
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
"""Raster normalisation routine."""

import warnings

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtWidgets

from pygmi.misc import BasicModule
from pygmi.raster.datatypes import Data
from pygmi.raster.misc import histeq

warnings.simplefilter("always", RuntimeWarning)


class Normalisation(BasicModule):
    """
    Class Normalisation GUI.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.rb_interval = QtWidgets.QRadioButton("Interval [0 1]")
        self.rb_mean = QtWidgets.QRadioButton("Mean: zero,  Standard deviation: unity")
        self.rb_median = QtWidgets.QRadioButton(
            "Median: zero,  Median absolute deviation: unity"
        )
        self.rb_8bit = QtWidgets.QRadioButton("8-bit histogram equalisation [0 255]")

        self.setupui()

    def setupui(self):
        """Set up UI."""
        vbl_1 = QtWidgets.QVBoxLayout(self)

        self.buttonbox.htmlfile = "raster.dm.norm"

        self.rb_interval.setChecked(True)

        gbox = QtWidgets.QGroupBox("Normalisation/Scaling")
        vbl_2 = QtWidgets.QVBoxLayout(gbox)

        vbl_2.addWidget(self.rb_interval)
        vbl_2.addWidget(self.rb_mean)
        vbl_2.addWidget(self.rb_median)
        vbl_2.addWidget(self.rb_8bit)

        vbl_1.addWidget(gbox)
        vbl_1.addWidget(self.buttonbox)

        self.setWindowTitle("Normalisation")

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
        if "Raster" not in self.indata:
            self.showlog("No Raster Data.")
            return False

        if not nodialog:
            temp = self.exec()
            if temp == 0:
                return False

        data = [i.copy() for i in self.indata["Raster"]]

        ntype = "interval"
        if self.rb_interval.isChecked():
            ntype = "interval"
        elif self.rb_mean.isChecked():
            ntype = "mean"
        elif self.rb_median.isChecked():
            ntype = "median"
        elif self.rb_8bit.isChecked():
            ntype = "8bit"

        # Correct the null value
        for i in data:
            i.data.data[i.data.mask] = i.nodata

        data = norm(data, ntype)

        self.outdata["Raster"] = data
        if self.pbar is not None:
            self.pbar.to_max()
        return True

    def saveproj(self):
        """Save project data from class."""
        self.saveobj(self.rb_interval)
        self.saveobj(self.rb_mean)
        self.saveobj(self.rb_median)
        self.saveobj(self.rb_8bit)


def datacommon(data: Data, tmp1: float, tmp2: float) -> tuple[Data, NDArray]:
    """
    Variables used in the process routine.

    Parameters
    ----------
    data
        PyGMI raster dataset.
    tmp1
        Parameter 1. Can be min, mean or median of the data.
    tmp2
        Parameter 2. Can be range, std, or Median Absolute Deviation (MAD) of
        the data.

    Returns
    -------
    data : Data
        PyGMI raster dataset.
    transform : ndarray.
        Transformation applied to data.

    """
    transform = np.zeros((2, 2))
    if tmp1 != 0.0 or tmp2 != 1.0:
        transform[0:2, 0] = [0, 1]
        transform[0:2, 1] = [tmp1, tmp2]

        dtmp = data.data.data
        mtmp = data.data.mask
        dtmp -= tmp1
        dtmp /= tmp2

        data.data = np.ma.array(dtmp, mask=mtmp)

    return data, transform


def norm(data: list[Data], ntype: str) -> list[Data]:
    """
    Normalise data.

    Parameters
    ----------
    data
        PyGMI Data in a list.
    ntype
        Normalisation type.Can be 'interval', 'mean', 'median' or '8bit'.

    Returns
    -------
    list of Data
        PyGMI Data in a list.

    """
    if ntype == "interval":
        for i in data:
            tmp1 = i.data.min()
            tmp2 = i.data.max() - i.data.min()
            i, _ = datacommon(i, tmp1, tmp2)
    elif ntype == "mean":
        for i in data:
            tmp1 = i.data.mean()
            tmp2 = i.data.std()
            i, _ = datacommon(i, tmp1, tmp2)
    elif ntype == "median":
        for i in data:
            tmp1 = np.median(i.data.compressed())
            tmp2 = np.median(abs(i.data.compressed() - tmp1))
            i, _ = datacommon(i, tmp1, tmp2)
    elif ntype == "8bit":
        for i in data:
            i.data = histeq(i.data)
            i.data = 255 * (i.data / np.ma.ptp(i.data))

    # Correct the null value
    for i in data:
        i.data.data[i.data.mask] = i.nodata

    return data


def _testfn():
    import sys

    from pygmi.raster.iodefs import get_raster

    ifile = r"D:\workdata\PyGMI Test Data\Raster\testdata.tif"

    dat = get_raster(ifile)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    DM = Normalisation()
    DM.indata["Raster"] = dat
    DM.settings()


if __name__ == "__main__":
    _testfn()
