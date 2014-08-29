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

# pylint: disable=E1101, C0103
from PyQt4 import QtGui, QtCore
import numpy as np
import scipy.stats as sstat
import copy
import warnings

warnings.simplefilter('always', RuntimeWarning)


class Normalisation(QtGui.QDialog):
    """ Class Normalisation """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.reportback = self.parent.showprocesslog

        self.verticallayout = QtGui.QVBoxLayout(self)
        self.groupbox = QtGui.QGroupBox(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.verticallayout_2 = QtGui.QVBoxLayout(self.groupbox)
        self.radiobutton_interval = QtGui.QRadioButton(self.groupbox)
        self.radiobutton_mean = QtGui.QRadioButton(self.groupbox)
        self.radiobutton_median = QtGui.QRadioButton(self.groupbox)
        self.radiobutton_8bit = QtGui.QRadioButton(self.groupbox)
        self.verticallayout.addWidget(self.groupbox)

        self.setupui()

        self.name = "Normalisation"
        self.normtype = 'minmax'  # mimax/meanstd/medmad/histeq

    def setupui(self):
        """ Setup UI """
#        self.resize(286, 166)
        self.radiobutton_interval.setChecked(True)
        self.verticallayout_2.addWidget(self.radiobutton_interval)
        self.radiobutton_mean = QtGui.QRadioButton(self.groupbox)
        self.verticallayout_2.addWidget(self.radiobutton_mean)
        self.verticallayout_2.addWidget(self.radiobutton_median)
        self.verticallayout_2.addWidget(self.radiobutton_8bit)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.verticallayout.addWidget(self.buttonbox)

        self.setWindowTitle("Normalisation")
        self.groupbox.setTitle("Normalisation/Scaling")
        self.radiobutton_interval.setText("Interval [0 1]")
        self.radiobutton_mean.setText(
            "Mean: zero,  Standard deviation: unity")
        self.radiobutton_median.setText(
            "Median: zero,  Median absolute deviation: unity")
        self.radiobutton_8bit.setText("8-bit histogram equalisation [0 255]")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)

    def settings(self):
        """ Settings """
        temp = self.exec_()
        if temp == 0:
            return

        data = copy.deepcopy(self.indata['Raster'])
#        if datachecks.Datachecks(self).isdata(data) is False:
#            return data
        transform = np.zeros((2, 2))
        if self.radiobutton_interval.isChecked():
            for i in data:
                tmp1 = i.data.min()
                tmp2 = i.data.max() - i.data.min()
                tmp3 = 'minmax'
                norm_type = 'Normalised to interval [0 1]'
                i, transform = self.datacommon(i, tmp1, tmp2, tmp3, norm_type)
        elif self.radiobutton_mean.isChecked():
            for i in data:
                tmp1 = i.data.mean()
                tmp2 = i.data.std()
                tmp3 = 'meanstd'
                norm_type = 'Normalised to mean, standard deviation = 1'
                i, transform = self.datacommon(i, tmp1, tmp2, tmp3, norm_type)
        elif self.radiobutton_median.isChecked():
            for i in data:
                tmp1 = np.median(i.data.compressed())
                tmp2 = np.median(abs(i.data.compressed() - tmp1))
                tmp3 = 'medmad'
                norm_type = 'Normalised to median, mean absolute deviation = 1'
                i, transform = self.datacommon(i, tmp1, tmp2, tmp3, norm_type)
        elif self.radiobutton_8bit.isChecked():
            for i in data:
                norm_type = 'Normalised-histogram equalisation'
                if True:
#                if (i.proc.__contains__(norm_type)) is False:
                    nlevels = 256
                    no_pix = i.data.count()
                    dummy_dat = np.sort(i.data[np.isnan(i.data) != 1],
                                        axis=None)

#                    ndat_eq = np.array(data[i].data, copy=True)
#                    ndat_eq = np.nan
                    ndat_eq = np.array(i.data, copy=True) * np.nan
                    transform = np.zeros((nlevels, 2))
#                    prog = wx.ProgressDialog('Histogram equalisation', \
#                        'Normalisation', nlevels, None, wx.PD_SMOOTH)
                    tmp = i.data.flatten('F')
                    tmpndat = ndat_eq.flatten('F')
                    for j in range(nlevels):
                        cop = np.round(float(j+1)*(no_pix/nlevels))
                        cutoff = dummy_dat[cop-1]
                        idx = np.nonzero(tmp < cutoff)
                        transform[j, 0:2] = [j+1, sstat.nanmedian(tmp[idx])]
                        tmp[idx] = np.nan
                        tmpndat[idx] = j+1
#                        prog.Update(j)
#                    prog.Destroy()
                    ndat_eq = np.reshape(tmpndat, i.data.shape, 'F')
                    tmpd = np.ma.array(ndat_eq)
                    i.data = tmpd.astype('float32')
                    i.data = np.ma.masked_invalid(i.data)
#                    i.proc.append(norm_type)
                    n_norms = len(i.norm)
                    i.norm[n_norms] = {'type': 'histeq',
                                       'transform': transform}

        self.outdata['Raster'] = data
        return True

    def datacommon(self, data, tmp1, tmp2, tmp3, tmp4):
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
#            data.data -= tmp1
#            data.data /= tmp2
#            data.data[np.isnan(data.data)] = data.nullvalue
            n_norms = len(data.norm)
#            data.proc.append(tmp4)
            data.norm[n_norms] = {'type': tmp3, 'transform': transform}
        return data, transform
