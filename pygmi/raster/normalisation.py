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
"""Normalisation function."""

import warnings
from PyQt5 import QtWidgets, QtCore
import numpy as np

from pygmi import menu_default
from pygmi.raster.ginterp import histeq
from pygmi.misc import BasicModule

warnings.simplefilter('always', RuntimeWarning)


class Normalisation(BasicModule):
    """Class Normalisation."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.rb_interval = QtWidgets.QRadioButton('Interval [0 1]')
        self.rb_mean = QtWidgets.QRadioButton('Mean: zero,  '
                                                       'Standard deviation: '
                                                       'unity')
        self.rb_median = QtWidgets.QRadioButton('Median: zero,  '
                                                         'Median absolute '
                                                         'deviation: unity')
        self.rb_8bit = QtWidgets.QRadioButton('8-bit histogram '
                                                       'equalisation [0 255]')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        vbl_1 = QtWidgets.QVBoxLayout(self)
        hbl = QtWidgets.QHBoxLayout()
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.normalisation')

        self.rb_interval.setChecked(True)

        gbox = QtWidgets.QGroupBox('Normalisation/Scaling')
        vbl_2 = QtWidgets.QVBoxLayout(gbox)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        vbl_2.addWidget(self.rb_interval)
        vbl_2.addWidget(self.rb_mean)
        vbl_2.addWidget(self.rb_median)
        vbl_2.addWidget(self.rb_8bit)

        hbl.addWidget(helpdocs)
        hbl.addWidget(buttonbox)

        vbl_1.addWidget(gbox)
        vbl_1.addLayout(hbl)

        self.setWindowTitle('Normalisation')

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
        if 'Raster' not in self.indata:
            self.showlog('No Raster Data.')
            return False

        if not nodialog:
            temp = self.exec()
            if temp == 0:
                return False

        data = [i.copy() for i in self.indata['Raster']]

        if self.rb_interval.isChecked():
            for i in data:
                tmp1 = i.data.min()
                tmp2 = i.data.max() - i.data.min()
                i, _ = datacommon(i, tmp1, tmp2)
        elif self.rb_mean.isChecked():
            for i in data:
                tmp1 = i.data.mean()
                tmp2 = i.data.std()
                i, _ = datacommon(i, tmp1, tmp2)
        elif self.rb_median.isChecked():
            for i in data:
                tmp1 = np.median(i.data.compressed())
                tmp2 = np.median(abs(i.data.compressed() - tmp1))
                i, _ = datacommon(i, tmp1, tmp2)
        elif self.rb_8bit.isChecked():
            for i in data:
                i.data = histeq(i.data)
                i.data = 255*(i.data/i.data.ptp())

        # Correct the null value
        for i in data:
            i.data.data[i.data.mask] = i.nodata

        self.outdata['Raster'] = data
        if self.pbar is not None:
            self.pbar.to_max()
        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.rb_interval)
        self.saveobj(self.rb_mean)
        self.saveobj(self.rb_median)
        self.saveobj(self.rb_8bit)


def datacommon(data, tmp1, tmp2):
    """
    Variables used in the process routine.

    Parameters
    ----------
    data : PyGMI Data.
        PyGMI raster dataset.
    tmp1 : float
        Parameter 1. Can be min, mean or median.
    tmp2 : float
        Parameter 2. Can be range, std, or mad.

    Returns
    -------
    data : PyGMI Data
        PyGMI raster dataset.
    transform : numpy array.
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
