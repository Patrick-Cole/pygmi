# -----------------------------------------------------------------------------
# Name:        smooth.py (part of PyGMI)
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
""" Smooth Data """

from PyQt4 import QtGui
import numpy as np
import scipy.signal as ssig
from scipy.stats import mode
import copy


class Smooth(QtGui.QDialog):
    """ Smooth """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.reportback = self.parent.showprocesslog

        self.gridlayout = QtGui.QGridLayout(self)
        self.groupbox = QtGui.QGroupBox(self)
        self.gridlayout_2 = QtGui.QGridLayout(self.groupbox)
        self.label = QtGui.QLabel(self.groupbox)
        self.spinbox_x = QtGui.QSpinBox(self.groupbox)
        self.label_2 = QtGui.QLabel(self.groupbox)
        self.label_3 = QtGui.QLabel(self.groupbox)
        self.spinbox_y = QtGui.QSpinBox(self.groupbox)
        self.spinbox_radius = QtGui.QSpinBox(self.groupbox)
        self.label_4 = QtGui.QLabel(self.groupbox)
        self.spinbox_stddev = QtGui.QSpinBox(self.groupbox)
        self.groupbox_2 = QtGui.QGroupBox(self)
        self.verticallayout = QtGui.QVBoxLayout(self.groupbox_2)
        self.radiobutton_2dmode = QtGui.QRadioButton(self.groupbox_2)
        self.verticallayout.addWidget(self.radiobutton_2dmode)
        self.radiobutton_2dmedian = QtGui.QRadioButton(self.groupbox_2)
        self.verticallayout.addWidget(self.radiobutton_2dmedian)
        self.radiobutton_2dmean = QtGui.QRadioButton(self.groupbox_2)
        self.verticallayout.addWidget(self.radiobutton_2dmean)
        self.groupbox_3 = QtGui.QGroupBox(self)
        self.verticallayout_2 = QtGui.QVBoxLayout(self.groupbox_3)
        self.radiobutton_box = QtGui.QRadioButton(self.groupbox_3)
        self.verticallayout_2.addWidget(self.radiobutton_box)
        self.radiobutton_disk = QtGui.QRadioButton(self.groupbox_3)
        self.verticallayout_2.addWidget(self.radiobutton_disk)
        self.radiobutton_gaussian = QtGui.QRadioButton(self.groupbox_3)
        self.verticallayout_2.addWidget(self.radiobutton_gaussian)
        self.tablewidget = QtGui.QTableWidget(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.progressbar = QtGui.QProgressBar(self)

        self.setupui()

        self.fmat = None
# options : 2dmean, 2dmode, 2dmedian
#        self.filtertype = "2D Median"
# options :'box', 'disc' or 'gaussian' (2dmean only)
#        self.filtershape = "Box Window"
#        self.filtersize_x = 5
#        self.filtersize_y = 5
#        self.filtersize_radius = 5
#        self.gaussfilterstdev = 5
#        self.name = "Smoothing"
#        show_xy = True
#        show_rad = False
#        showstddev = False
#        show_gauss = False

        self.radiobutton_2dmean.clicked.connect(self.choosefilter)
        self.radiobutton_2dmedian.clicked.connect(self.choosefilter)
        self.radiobutton_2dmode.clicked.connect(self.choosefilter)
        self.radiobutton_box.clicked.connect(self.choosefilter)
        self.radiobutton_disk.clicked.connect(self.choosefilter)
        self.radiobutton_gaussian.clicked.connect(self.choosefilter)
        self.spinbox_x.valueChanged.connect(self.choosefilter)
        self.spinbox_y.valueChanged.connect(self.choosefilter)
        self.spinbox_radius.valueChanged.connect(self.choosefilter)
        self.spinbox_stddev.valueChanged.connect(self.choosefilter)

        self.choosefilter()

    def setupui(self):
        """ Setup UI """
        self.gridlayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridlayout_2.addWidget(self.spinbox_x, 0, 1, 1, 1)
        self.gridlayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.gridlayout_2.addWidget(self.label_3, 2, 0, 1, 1)
        self.gridlayout_2.addWidget(self.label_4, 3, 0, 1, 1)
        self.gridlayout_2.addWidget(self.spinbox_y, 1, 1, 1, 1)
        self.gridlayout_2.addWidget(self.spinbox_radius, 2, 1, 1, 1)
        self.gridlayout_2.addWidget(self.spinbox_stddev, 3, 1, 1, 1)
        self.gridlayout.addWidget(self.tablewidget, 1, 0, 1, 3)
        self.gridlayout.addWidget(self.groupbox_2, 2, 0, 1, 1)
        self.gridlayout.addWidget(self.progressbar, 3, 0, 1, 3)
        self.gridlayout.addWidget(self.groupbox_3, 2, 1, 1, 1)
        self.gridlayout.addWidget(self.buttonbox, 4, 1, 1, 1)
        self.gridlayout.addWidget(self.groupbox, 2, 2, 1, 1)

        self.spinbox_x.setMinimum(1)
        self.spinbox_x.setMaximum(999999)
        self.spinbox_x.setProperty("value", 5)
        self.spinbox_y.setMinimum(1)
        self.spinbox_y.setMaximum(9999999)
        self.spinbox_y.setProperty("value", 5)
        self.spinbox_radius.setMinimum(1)
        self.spinbox_radius.setMaximum(99999)
        self.spinbox_radius.setProperty("value", 5)
        self.spinbox_stddev.setMinimum(1)
        self.spinbox_stddev.setMaximum(99999)
        self.spinbox_stddev.setProperty("value", 5)
        self.radiobutton_2dmode.setChecked(True)
        self.radiobutton_box.setChecked(True)
        self.tablewidget.setRowCount(5)
        self.tablewidget.setColumnCount(5)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.progressbar.setProperty("value", 0)

        self.setWindowTitle("Smoothing Filters")
        self.groupbox.setTitle("Filter Size")
        self.label.setText("X:")
        self.label_2.setText("Y:")
        self.label_3.setText("Radius in Samples:")
        self.label_4.setText("Standard Deviation:")
        self.groupbox_2.setTitle("Filter Type")
        self.radiobutton_2dmode.setText("2D Mode")
        self.radiobutton_2dmedian.setText("2D Median")
        self.radiobutton_2dmean.setText("2D Mean")
        self.groupbox_3.setTitle("Filter Shape")
        self.radiobutton_box.setText("Box Window")
        self.radiobutton_disk.setText("Disk Window")
        self.radiobutton_gaussian.setText("Gaussian Window")

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    def settings(self):
        """ Settings """
        temp = self.exec_()
        if temp == 0:
            return

        self.parent.process_is_active(True)
        self.parent.showprocesslog('Smoothing ')
        data = copy.deepcopy(self.indata['Raster'])
#        if datachecks.Datachecks(self).isdata(data) is False:
#            return data
#        if datachecks.Datachecks(self).isnorm(data) is True:
#            return data
        if self.radiobutton_2dmean.isChecked():
            for i in range(len(data)):
                data[i].data = self.mov_win_filt(data[i].data, self.fmat,
                                                 '2D Mean', data[i].dataid)

        if self.radiobutton_2dmedian.isChecked():
            for i in range(len(data)):
                data[i].data = self.mov_win_filt(data[i].data, self.fmat,
                                                 '2D Median', data[i].dataid)

        if self.radiobutton_2dmode.isChecked():
            for i in range(len(data)):
                data[i].data = self.mov_win_filt(data[i].data, self.fmat,
                                                 '2D Mode', data[i].dataid)

        self.parent.process_is_active(False)
        self.outdata['Raster'] = data
        self.parent.showprocesslog('Finished!', True)

        return True

    def choosefilter(self):
        """ Section to choose the filter """
        # Do not need to check whether inputs are greater than zero,
        # since the spinboxes will not permit it.

        box_x = self.spinbox_x.value()
        box_y = self.spinbox_x.value()
        rad = self.spinbox_radius.value()
        sigma = self.spinbox_stddev.value()
        fmat = None

        if self.radiobutton_2dmean.isChecked():
            self.radiobutton_gaussian.setVisible(True)
            if self.radiobutton_box.isChecked():
                fmat = filters2d('average', [box_y, box_x])
            elif self.radiobutton_disk.isChecked():
                fmat = filters2d('disc', rad)
            elif self.radiobutton_gaussian.isChecked():
                fmat = filters2d('gaussian', [box_x, box_y], sigma)
        else:
            self.radiobutton_gaussian.setVisible(False)
            if self.radiobutton_gaussian.isChecked():
                self.radiobutton_box.setChecked(True)
            if self.radiobutton_box.isChecked():
                fmat = np.ones((box_y, box_x))
            elif self.radiobutton_disk.isChecked():
                fmat = filters2d('disc', rad)
                fmat[fmat > 0] = 1

        if self.radiobutton_box.isChecked():
            self.spinbox_x.setVisible(True)
            self.spinbox_y.setVisible(True)
            self.spinbox_radius.setVisible(False)
            self.spinbox_stddev.setVisible(False)
            self.label.setVisible(True)
            self.label_2.setVisible(True)
            self.label_3.setVisible(False)
            self.label_4.setVisible(False)
        elif self.radiobutton_disk.isChecked():
            self.spinbox_x.setVisible(False)
            self.spinbox_y.setVisible(False)
            self.spinbox_radius.setVisible(True)
            self.spinbox_stddev.setVisible(False)
            self.label.setVisible(False)
            self.label_2.setVisible(False)
            self.label_3.setVisible(True)
            self.label_4.setVisible(False)
        elif self.radiobutton_gaussian.isChecked():
            self.spinbox_x.setVisible(False)
            self.spinbox_y.setVisible(False)
            self.spinbox_radius.setVisible(False)
            self.spinbox_stddev.setVisible(True)
            self.label.setVisible(False)
            self.label_2.setVisible(False)
            self.label_3.setVisible(False)
            self.label_4.setVisible(True)

        self.fmat = fmat
        self.updatetable()

    def updatetable(self):
        """ Update Table """
        if self.fmat is None:
            return

        red = [0]*85 + list(range(0, 256, 3))+[255]*85
        green = list(range(0, 256, 3)) + 85*[255] + list(range(255, 0, -3))
        blue = [255]*85 + list(range(255, 0, -3)) + [0]*86
        fmin = self.fmat.min()
        frange = self.fmat.ptp()

        self.tablewidget.setRowCount(self.fmat.shape[0])
        self.tablewidget.setColumnCount(self.fmat.shape[1])

        for row in range(self.fmat.shape[0]):
            for col in range(self.fmat.shape[1]):
                if frange == 0:
                    i = 127
                else:
                    i = int(255*(self.fmat[row, col]-fmin)/frange)
                ltmp = QtGui.QLabel('{:g}'.format(self.fmat[row, col]))
                ltmp.setStyleSheet(
                    'Background: rgb' +
                    str((red[i], green[i], blue[i])) + '; Color: rgb' +
                    str((255-red[i], 255-green[i], 255-blue[i])))
                self.tablewidget.setCellWidget(row, col, ltmp)

        self.tablewidget.resizeColumnsToContents()

    def msgbox(self, title, message):
        """ Msgbox """
        QtGui.QMessageBox.warning(self.parent, title, message,
                                  QtGui.QMessageBox.Ok, QtGui.QMessageBox.Ok)

    def mov_win_filt(self, dat, fmat, itype, title):
        """ move win filt function """
        out = dat.tolist()
#        self.progressbar.setValue(0)

        rowf = fmat.shape[0]
        colf = fmat.shape[1]
        rowd = dat.shape[0]
        cold = dat.shape[1]
        drr = round(rowf/2.0)
        dcc = round(colf/2.0)

        dummy = np.zeros((rowd+rowf-1, cold+colf-1))*np.nan
        dummy[drr-1:drr-1+rowd, dcc-1:dcc-1+cold] = dat
        dummy = np.ma.masked_invalid(dummy)
        dummy.mask[drr-1:drr-1+rowd, dcc-1:dcc-1+cold] = dat.mask

        dat.data[dat.mask == 1] = np.nan

        if itype == '2D Mean':
            out = ssig.correlate(dat, fmat, 'same')
            self.progressbar.setValue(100)

        else:
            fmat[fmat != 1] = np.nan
            out = []
            for i in range(rowd):
                self.parent.showprocesslog(title+' Progress: ' +
                                           str(round(100*i/rowd))+'%', True)
#                self.progressbar.setValue(100*i/rowd)
                out.append([dummy[i:i+rowf, j:j+colf]
                            for j in range(cold)])
            fmatflat = fmat.flatten()
            out = np.array(out)
            out.shape = (rowd, cold, rowf*colf)
            out = out*fmatflat
            if itype == '2D Median':
                self.parent.showprocesslog('Calculating Median...')
                out = np.median(out, 2)
            if itype == '2D Mode':
                self.parent.showprocesslog('Calculating Mode...')
                out = mode(out, 2)[0]

        out = np.ma.masked_invalid(out)
        out.shape = out.shape[0:2]
        out.mask[:rowf/2] = True
        out.mask[-rowf/2:] = True
        out.mask[:, :colf/2] = True
        out.mask[:, -colf/2:] = True
        return out


def filters2d(filtertype, sze, *sigma):
    """ Filters 2D

    These filter definitions have been translated from the octave function
    'fspecial'.

    Args:
        filtertype (str): Type of filter. Can be 'average', 'disc' or
            'gaussian'.
        sze (numpy array or integer): This is a integer radius for 'disc' or a
            vector containing rows and columns otherwize.
        sigma (numpy array): numpy array containing std deviation. Used in
            'gaussian'

    Return:
        numpy array: Returns the filter to be used.

    Copyright (C) 2005 Peter Kovesi

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

    FSPECIAL - Create spatial filters for image processing

    Usage:  f = fspecial(filtertype, optionalparams)

    filtertype can be

|       'average'   - Rectangular averaging filter
|       'disc'      - Circular averaging filter.
|       'gaussian'  - Gaussian filter.

    The parameters that need to be specified depend on the filtertype

    Examples of use and associated default values:

|     f = fspecial('average',sze)           sze can be a 1 or 2 vector
|                                            default is [3 3].
|     f = fspecial('disk',radius)           default radius = 5
|     f = fspecial('gaussian',sze, sigma)   default sigma is 0.5

    Where sze is specified as a single value the filter will be square.

|    Author:   Peter Kovesi <pk@csse.uwa.edu.au>
|    Keywords: image processing, spatial filters
|    Created:  August 2005
    """

    if filtertype == 'disc':
        radius = sze
        sze = [2*radius+1, 2*radius+1]

    rows = sze[0]
    cols = sze[1]
    r2 = (rows-1)/2
    c2 = (cols-1)/2

    if filtertype == 'average':
        rows = sze[0]
        cols = sze[1]
        f = np.ones(sze)/(rows*cols)

    elif filtertype == 'disc':
        [x, y] = np.mgrid[-c2: c2, -r2: r2]
        rad = np.sqrt(x**2 + y**2)
        f = rad <= radius
        f = f/np.sum(f[:])

    elif filtertype == 'gaussian':
        [x, y] = np.mgrid[-c2: c2, -r2:r2]
        radsqrd = x**2 + y**2
        f = np.exp(-radsqrd/(2*sigma[0]**2))
        f = f/np.sum(f[:])
    else:
        print('Unrecognized filter type')

    return f
