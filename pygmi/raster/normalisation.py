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
""" Normalisation function """

import copy
import warnings
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pygmi.menu_default as menu_default


warnings.simplefilter('always', RuntimeWarning)


class Normalisation(QtWidgets.QDialog):
    """ Class Normalisation """
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = parent.pbar
        self.reportback = self.parent.showprocesslog

        self.radiobutton_interval = QtWidgets.QRadioButton('Interval [0 1]')
        self.radiobutton_mean = QtWidgets.QRadioButton('Mean: zero,  Standard deviation: unity')
        self.radiobutton_median = QtWidgets.QRadioButton('Median: zero,  Median absolute deviation: unity')
        self.radiobutton_8bit = QtWidgets.QRadioButton('8-bit histogram equalisation [0 255]')

        self.setupui()

        self.name = 'Normalisation'
        self.normtype = 'minmax'  # mimax/meanstd/medmad/histeq

    def setupui(self):
        """ Setup UI """

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

    def settings(self):
        """ Settings """
        temp = self.exec_()
        if temp == 0:
            return False

        data = copy.deepcopy(self.indata['Raster'])
        transform = np.zeros((2, 2))
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
                nlevels = 256
                no_pix = i.data.count()
                dummy_dat = np.sort(i.data[np.isnan(i.data) != 1],
                                    axis=None)

                ndat_eq = np.array(i.data, copy=True) * np.nan
                transform = np.zeros((nlevels, 2))
                tmp = i.data.flatten('F')
                tmpndat = ndat_eq.flatten('F')
                for j in range(nlevels):
                    cop = np.int(np.round(float(j+1)*(no_pix/nlevels)))
                    cutoff = dummy_dat[cop-1]
                    idx = np.nonzero(tmp < cutoff)
                    transform[j, 0:2] = [j+1, np.nanmedian(tmp[idx])]
                    tmp[idx] = np.nan
                    tmp = np.ma.fix_invalid(tmp)
                    tmpndat[idx] = j+1
                ndat_eq = np.reshape(tmpndat, i.data.shape, 'F')
                tmpd = np.ma.array(ndat_eq)
                i.data = tmpd.astype('float32')
                i.data = np.ma.masked_invalid(i.data)
#                n_norms = len(i.norm)
#                i.norm[n_norms] = {'type': 'histeq',
#                                   'transform': transform}

        # Correct the null value
        for i in data:
            i.data.data[i.data.mask] = i.nullvalue

        self.outdata['Raster'] = data
        self.pbar.to_max()
        return True


def datacommon(data, tmp1, tmp2, tmp3):
    """ Common stuff used in the process routine """
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
