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

import copy
import warnings
from PyQt5 import QtWidgets, QtCore
import numpy as np

import pygmi.menu_default as menu_default
from pygmi.raster.ginterp import histeq


warnings.simplefilter('always', RuntimeWarning)


class Normalisation(QtWidgets.QDialog):
    """Class Normalisation."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        if parent is not None:
            self.pbar = parent.pbar
        else:
            self.pbar = None

        self.radiobutton_interval = QtWidgets.QRadioButton('Interval [0 1]')
        self.radiobutton_mean = QtWidgets.QRadioButton('Mean: zero,  '
                                                       'Standard deviation: '
                                                       'unity')
        self.radiobutton_median = QtWidgets.QRadioButton('Median: zero,  '
                                                         'Median absolute '
                                                         'deviation: unity')
        self.radiobutton_8bit = QtWidgets.QRadioButton('8-bit histogram '
                                                       'equalisation [0 255]')

        self.setupui()

        self.normtype = 'minmax'  # mimax/meanstd/medmad/histeq

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        verticallayout = QtWidgets.QVBoxLayout(self)
        horizontallayout = QtWidgets.QHBoxLayout()
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.normalisation')

        self.radiobutton_interval.setChecked(True)

        groupbox = QtWidgets.QGroupBox('Normalisation/Scaling')
        verticallayout_2 = QtWidgets.QVBoxLayout(groupbox)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        verticallayout_2.addWidget(self.radiobutton_interval)
        verticallayout_2.addWidget(self.radiobutton_mean)
        verticallayout_2.addWidget(self.radiobutton_median)
        verticallayout_2.addWidget(self.radiobutton_8bit)

        horizontallayout.addWidget(helpdocs)
        horizontallayout.addWidget(buttonbox)

        verticallayout.addWidget(groupbox)
        verticallayout.addLayout(horizontallayout)

        self.setWindowTitle('Normalisation')

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if not nodialog:
            temp = self.exec_()
            if temp == 0:
                return False

        data = copy.deepcopy(self.indata['Raster'])
        # transform = np.zeros((2, 2))
        if self.radiobutton_interval.isChecked():
            for i in data:
                tmp1 = i.data.min()
                tmp2 = i.data.max() - i.data.min()
                tmp3 = 'minmax'
                i, transform = datacommon(i, tmp1, tmp2, tmp3)
        elif self.radiobutton_mean.isChecked():
            for i in data:
                tmp1 = i.data.mean()
                tmp2 = i.data.std()
                tmp3 = 'meanstd'
                i, transform = datacommon(i, tmp1, tmp2, tmp3)
        elif self.radiobutton_median.isChecked():
            for i in data:
                tmp1 = np.median(i.data.compressed())
                tmp2 = np.median(abs(i.data.compressed() - tmp1))
                tmp3 = 'medmad'
                i, transform = datacommon(i, tmp1, tmp2, tmp3)
        elif self.radiobutton_8bit.isChecked():
            for i in data:
                i.data = histeq(i.data)
                i.data = 255*(i.data/i.data.ptp())

        # Correct the null value
        for i in data:
            i.data.data[i.data.mask] = i.nullvalue

        self.outdata['Raster'] = data
        if self.pbar is not None:
            self.pbar.to_max()
        return True

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
        if projdata['type'] == 'interval':
            self.radiobutton_interval.setChecked(True)
        elif projdata['type'] == 'mean':
            self.radiobutton_mean.setChecked(True)
        elif projdata['type'] == 'median':
            self.radiobutton_median.setChecked(True)
        elif projdata['type'] == '8bit':
            self.radiobutton_8bit.setChecked(True)

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

        if self.radiobutton_interval.isChecked():
            projdata['type'] = 'interval'
        elif self.radiobutton_mean.isChecked():
            projdata['type'] = 'mean'
        elif self.radiobutton_median.isChecked():
            projdata['type'] = 'median'
        elif self.radiobutton_8bit.isChecked():
            projdata['type'] = '8bit'

        return projdata


def datacommon(data, tmp1, tmp2, tmp3):
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
    tmp3 : str
        Text label. Can be 'minmax', 'meanstd' or 'medmad'.

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
#        n_norms = len(data.norm)
#        data.norm[n_norms] = {'type': tmp3, 'transform': transform}
    return data, transform
